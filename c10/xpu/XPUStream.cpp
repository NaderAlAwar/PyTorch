#include <c10/util/CallOnce.h>
#include <c10/util/flat_hash_map.h>
#include <c10/util/irange.h>
#include <c10/xpu/XPUException.h>
#include <c10/xpu/XPUStream.h>

#include <atomic>
#include <deque>
#include <mutex>
#include <vector>

namespace c10::xpu {
namespace {

// Global stream state and constants
c10::once_flag init_flag;
DeviceIndex num_gpus = -1;
constexpr int kStreamsPerPoolBits = 5;
constexpr int kStreamsPerPool = 1 << kStreamsPerPoolBits;
constexpr int kStreamTypeBits = 3;

// The number of compile-time external streams is limited to twice the number of
// native streams. For more details, see Note [External XPU Stream].
constexpr int max_compile_time_external_streams =
    2 * kStreamsPerPool * max_compile_time_stream_priorities;

// The SYCL queue pools are lazily initialized when the first queue is requested
// for a device. The device flags track the initialization of each device. When
// a queue is requested, the next queue in the pool to be returned in a
// round-robin fashion, see Note [Stream Management].
std::deque<c10::once_flag> device_flags;
std::vector<std::array<
    std::array<std::unique_ptr<sycl::queue>, kStreamsPerPool>,
    max_compile_time_stream_priorities>>
    streams;
std::deque<
    std::array<std::atomic<uint32_t>, max_compile_time_stream_priorities>>
    priority_counters;

/*
 * Note [External XPU Stream]
 *
 * An external XPUStream is a wrapper around an external SYCL queue that was not
 * created by PyTorch. This design enables interoperability with other libraries
 * by allowing PyTorch to utilize SYCL queues created outside of its control.
 *
 * To achieve this, we need a robust mechanism to convert an external SYCL queue
 * into an XPUStream. Since a SYCL queue is effectively a handle to an
 * underlying queue implementation, multiple SYCL queue objects can reference
 * the same underlying queue. Therefore, we use the SYCL queue (not the raw
 * pointer) as the input to `getStreamFromExternal`.
 *
 * Key requirements for this design are:
 *   1. Fetching the external SYCL queue from the external XPUStream.
 *   2. Converting an external XPUStream to a `c10::Stream` and vice versa.
 *   3. Ensuring the external XPUStream supports `get/setCurrentXPUStream`.
 *   4. Enable memory caching allocation through the external XPUStream.
 *
 * To meet these requirements, we need to record the external SYCL queue pointer
 * using the `stream_id`. However, SYCL queue pointers can become invalid if the
 * referenced SYCL queue goes out of scope before the XPUStream. Likewise, any
 * block allocated by the external XPUStream will become invalid once the
 * associated XPUStream is destructed.
 *
 * To ensure the validity of SYCL queue pointer throughout the lifetime of an
 * XPUStream, we maintain an external queue mapping in `external_streams`. This
 * mapping tracks external queues and their associated pointers, ensuring they
 * remain valid as long as there is an active reference to the corresponding
 * XPUStream.
 *
 * However, due to the key requirements (2), (3), and (4), we CANNOT determine
 * when it is safe to release the external queue and its pointer from the map
 * `external_streams`. To address this, we keep the external queue persistently
 * alive in the pool, ensuring the SYCL queue pointer remains valid regardless
 * of whether the XPUStream is still being referenced.
 *
 * To prevent performance degradation, excessive memory usage, and increased
 * system complexity, the number of external streams at compile-time is limited
 * to twice the number of native streams. We assume that other libraries will
 * carefully manage the number of external streams to ensure efficient and
 * consistent behavior.
 */
std::vector<ska::flat_hash_map<sycl::queue, std::unique_ptr<sycl::queue>>>
    external_streams;
std::deque<std::mutex> external_stream_mutexs;

thread_local std::unique_ptr<StreamId[]> current_streams = nullptr;

// Note [StreamId assignment]
// ~~~~~~~~~~~~~~~~~~~~~~~~~~
// How do we assign stream IDs?
//
// -- 55 bits --    -- 5 bits --     -- 3 bits --     -- 1 bit --
//     zeros       StreamIdIndex     StreamIdType    Ext/native stream
//                ignored for ext   ignored for ext
//
// Where StreamIdType:
//  000 = normal priority queue
//  001 = high priority queue
//  111 = external queue
//
// StreamId is 64-bit, so we can just rely on regular promotion rules.
// We rely on StreamIdIndex and StreamIdType being non-negative;

using StreamIdIndex = uint8_t;
enum class StreamIdType : uint8_t {
  // The higher the type number, the higher the priority for the native stream.
  NORMAL = 0x0,
  HIGH = 0X1,
  // For an external stream, the last bit of StreamId is 0, whose priority is
  // queried at runtime.
  EXT = 0X7,
};

inline std::ostream& operator<<(std::ostream& stream, StreamIdType q) {
  switch (q) {
    case StreamIdType::NORMAL:
      return stream << "NORMAL";
    case StreamIdType::HIGH:
      return stream << "HIGH";
    case StreamIdType::EXT:
      return stream << "EXT";
    default:
      break;
  }
  return stream << static_cast<int16_t>(q);
}

inline StreamIdType streamIdType(StreamId s) {
  // Externally allocated streams have their id being the sycl::queue pointer
  // so the last bit will be 0
  if ((!(s & 1))) {
    return StreamIdType::EXT;
  }
  // The stream type mask starts from the second rightmost bit.
  int mask_for_type = (1 << kStreamTypeBits) - 1;
  auto st = static_cast<StreamIdType>((s >> 1) & mask_for_type);
  TORCH_CHECK(
      st == StreamIdType::NORMAL || st == StreamIdType::HIGH,
      "invalid StreamId: ",
      s);
  return st;
}

inline StreamIdIndex streamIdIndex(StreamId s) {
  // The stream index mask starts from the fourth rightmost bit.
  return static_cast<StreamIdIndex>(
      (s >> (kStreamTypeBits + 1)) & ((1 << kStreamsPerPoolBits) - 1));
}

inline StreamId makeStreamId(StreamIdType st, StreamIdIndex si) {
  return (static_cast<StreamId>(si) << (kStreamTypeBits + 1)) |
      (static_cast<StreamId>(st) << 1) | 1;
}

void initGlobalStreamState() {
  num_gpus = c10::xpu::device_count();
  device_flags.resize(num_gpus);
  streams.resize(num_gpus);
  priority_counters.resize(num_gpus);
  external_streams.resize(num_gpus);
  external_stream_mutexs.resize(num_gpus);
}

// Creates the reserved SYCL queue pools for the specified device. It should be
// call only once.
void initDeviceStreamState(DeviceIndex device) {
  using namespace sycl::ext::oneapi::property;
  // Need to align with StreamIdType.
  const std::vector<sycl::property_list> properties = {
      {sycl::property::queue::in_order(), queue::priority_normal()},
      {sycl::property::queue::in_order(), queue::priority_high()}};
  for (const auto p : c10::irange(max_compile_time_stream_priorities)) {
    for (const auto i : c10::irange(kStreamsPerPool)) {
      auto& stream = streams[device][p][i];
      stream = std::make_unique<sycl::queue>(sycl::queue(
          c10::xpu::get_device_context(),
          c10::xpu::get_raw_device(device),
          c10::xpu::asyncHandler,
          properties[p]));
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_stream_creation(
            c10::kXPU, reinterpret_cast<uintptr_t>(stream.get()));
      }
    }
    priority_counters[device][p] = 0;
  }
}

void initXPUStreamsOnce() {
  c10::call_once(init_flag, initGlobalStreamState);

  if (current_streams) {
    return;
  }

  // Inits current streams (thread local) to the last queue in the "normal
  // priority" queue pool. Note: the queue pool have not been initialized yet.
  // It will be initialized in initDeviceStreamState for the specified device.
  current_streams = std::make_unique<StreamId[]>(num_gpus);
  for (const auto i : c10::irange(num_gpus)) {
    // Assigning the current stream to the last one in the pool can be
    // beneficial in certain scenarios, particularly when users initialize their
    // workload to perform computations with the current stream (the last one)
    // and utilize stream (the first one) from the pool for communication, it
    // allows for different streams to overlap in computation and communication.
    current_streams[i] =
        makeStreamId(StreamIdType::NORMAL, kStreamsPerPool - 1);
  }
}

// Creates the reserved sycl queue pools for the specified device to ensure
// initialization only occurs once.
inline void initDeviceStreamOnce(DeviceIndex device) {
  c10::call_once(device_flags[device], initDeviceStreamState, device);
}

uint32_t get_idx(std::atomic<uint32_t>& counter) {
  auto raw_idx = counter++;
  return raw_idx % kStreamsPerPool;
}

XPUStream XPUStreamForId(DeviceIndex device_index, StreamId stream_id) {
  return XPUStream(
      XPUStream::UNCHECKED,
      Stream(
          Stream::UNSAFE,
          c10::Device(DeviceType::XPU, device_index),
          stream_id));
}

// Retrieve the stream_id from an external SYCL queue. If the external queue is
// not found in the map, it will be added to the map. The stream_id is the raw
// pointer to the external SYCL queue stored in the map.
StreamId getExternalXPUStreamId(
    sycl::queue ext_queue,
    DeviceIndex device_index) {
  TORCH_CHECK(ext_queue.is_in_order(), "External SYCL queue must be in-order");
  auto& device_external_stream = external_streams[device_index];
  std::scoped_lock<std::mutex> lock(external_stream_mutexs[device_index]);

  // Check if the external queue already exists in the map
  auto it = device_external_stream.find(ext_queue);
  if (it != device_external_stream.end()) {
    return reinterpret_cast<StreamId>(it->second.get());
  }

  TORCH_CHECK(
      device_external_stream.size() < max_compile_time_external_streams,
      "The number of external SYCL queue on the machine exceeds the compile-time maximum limit (",
      max_compile_time_external_streams,
      "). Please increase the maximum number and recompile PyTorch.");

  // Add the external queue and its raw pointer to the map. For more details,
  // see Note [External XPU Stream].
  auto ext_queue_ptr = std::make_unique<sycl::queue>(ext_queue);
  auto ext_stream_id = reinterpret_cast<StreamId>(ext_queue_ptr.get());
  TORCH_CHECK(
      !(ext_stream_id & 1),
      "StreamId of external SYCL queue is not properly aligned");
  device_external_stream.emplace(ext_queue, std::move(ext_queue_ptr));
  return ext_stream_id;
}

} // anonymous namespace

int XPUStream::priority() const {
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  if (C10_UNLIKELY(st == StreamIdType::EXT)) {
    // Query external stream priority
    using namespace sycl::ext::oneapi::property;
    // Default priority for SYCL queue is normal.
    st = StreamIdType::NORMAL;
    if (queue().has_property<queue::priority_normal>()) {
      st = StreamIdType::NORMAL;
    } else if (queue().has_property<queue::priority_high>()) {
      st = StreamIdType::HIGH;
    }
  }
  // StreamIdType and priority number are inversely related.
  return -static_cast<int>(st);
}

// See Note [StreamId assignment]
sycl::queue& XPUStream::queue() const {
  DeviceIndex device_index = stream_.device_index();
  StreamId stream_id = stream_.id();
  StreamIdType st = streamIdType(stream_id);
  StreamIdIndex si = streamIdIndex(stream_id);
  switch (st) {
    case StreamIdType::NORMAL:
    case StreamIdType::HIGH:
      return *streams[device_index][static_cast<uint8_t>(st)][si];
    case StreamIdType::EXT:
      return *(reinterpret_cast<sycl::queue*>(stream_id));
    default:
      TORCH_CHECK(
          false,
          "Unrecognized stream ",
          stream_,
          " (I didn't recognize the stream type, ",
          st,
          ").",
          " Did you manufacture the StreamId yourself?  Don't do that;");
  }
}

// Returns a stream from the requested pool
// Note: The stream pools will be initialized if needed, at the first invocation
// to this function.
XPUStream getStreamFromPool(const int priority, DeviceIndex device) {
  initXPUStreamsOnce();
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device_index(device);
  TORCH_CHECK(
      priority <= 0,
      "Expected XPU stream priority to be less than or equal to 0, got ",
      priority);
  // Initializes the stream pools (once)
  initDeviceStreamOnce(device);
  auto priority_idx =
      std::min(-priority, max_compile_time_stream_priorities - 1);
  const auto idx = get_idx(priority_counters[device][priority_idx]);
  auto id_type = static_cast<StreamIdType>(priority_idx);
  return XPUStreamForId(device, makeStreamId(id_type, idx));
}

XPUStream getStreamFromPool(const bool isHighPriority, DeviceIndex device) {
  initXPUStreamsOnce();
  // If isHighPriority is true, return the stream with the highest priority.
  int priority = isHighPriority ? -max_compile_time_stream_priorities + 1 : 0;
  return getStreamFromPool(priority, device);
}

// Note: The stream pools will be initialized if needed, at the first invocation
// to this function.
XPUStream getCurrentXPUStream(DeviceIndex device) {
  initXPUStreamsOnce();
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device_index(device);
  // Initializes the stream pool (once)
  initDeviceStreamOnce(device);
  return XPUStreamForId(device, current_streams[device]);
}

// Note: The stream pools will be initialized if needed, at the first invocation
// to this function.
void setCurrentXPUStream(XPUStream stream) {
  initXPUStreamsOnce();
  current_streams[stream.device_index()] = stream.id();
}

std::ostream& operator<<(std::ostream& stream, const XPUStream& s) {
  return stream << s.unwrap();
}

/*
 * Note [Synchronize Streams on Device]
 *
 * There are two stream pools per device to manage our reserved SYCL queues.
 * When syncStreamsOnDevice is called, all reserved SYCL queues in the pools of
 * the specified device will be blocked, and wait for their synchronizations. We
 * realize the semantics via a loop through the stream pools of the specified
 * device and make each command queue synchronization sequentially.
 *
 * There is a semantic gap with device synchronization because only the SYCL
 * queues we have reserved (in our pools) will be synchronized, rather than
 * synchronizing all SYCL queues on the specified device.
 */

// Note: The stream pools will be initialized if needed, at the first invocation
// to this function.
void syncStreamsOnDevice(DeviceIndex device) {
  initXPUStreamsOnce();
  if (device == -1) {
    device = c10::xpu::current_device();
  }
  check_device_index(device);
  // Initializes the stream pools (once)
  initDeviceStreamOnce(device);

  // For each device, we have kStreamsPerPool (32) reserved queues per priority.
  for (const auto p : c10::irange(max_compile_time_stream_priorities)) {
    for (const auto i : c10::irange(kStreamsPerPool)) {
      streams[device][p][i]->wait();
    }
  }
  // For each device, we need to synchronize all external queues.
  for ([[maybe_unused]] const auto& [_, ext_queue] : external_streams[device]) {
    ext_queue->wait();
  }
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_device_synchronization(c10::kXPU);
  }
}

// Note: The external_streams pools will be initialized if needed, at the first
// invocation to this function.
XPUStream getStreamFromExternal(
    sycl::queue ext_queue,
    DeviceIndex device_index) {
  initXPUStreamsOnce();
  check_device_index(device_index);
  StreamId stream_id = getExternalXPUStreamId(ext_queue, device_index);
  return XPUStreamForId(device_index, stream_id);
}

} // namespace c10::xpu
