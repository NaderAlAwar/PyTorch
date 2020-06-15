#include <torch/csrc/cuda/nccl.h>
#include <torch/csrc/cuda/device_set.h>
#include <ATen/core/functional.h>
#include <torch/csrc/utils/hash.h>

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>

#include <THC/THC.h>

#include <nccl.h>

#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>

namespace torch {
namespace cuda {
namespace nccl {

using namespace at;

namespace detail {

void throw_nccl_error(torchNcclResult_t status) {
  std::ostringstream err;
  ncclResult_t nccl_status = static_cast<ncclResult_t>(status);
  err << "NCCL Error " << status << ": " << ncclGetErrorString(nccl_status);
  throw std::runtime_error(err.str());
}

 void NCCL_CHECK(torchNcclResult_t status) {
  if (status != torchNcclResult_t::ncclSuccess) {
    throw_nccl_error(status);
  }
}

struct NcclCommList {
  std::unique_ptr<ncclComm_t[]> comms;
  int ndevices;
  NcclCommList(const std::vector<int>& devices)
      : comms(new ncclComm_t[devices.size()]), ndevices(devices.size()) {
    NCCL_CHECK(
        static_cast<torchNcclResult_t>(ncclCommInitAll(
                          comms.get(), devices.size(), devices.data())));
  }
  NcclCommList(NcclCommList&& foo) = default;
  ~NcclCommList() {
    if (comms) {
      for (int i = 0; i < ndevices; i++) {
        int dummy_var;
        if (cudaGetDevice(&dummy_var) != cudaSuccess) {
          /* there are cases when this destructor is called after the
           CUDA driver is already unloaded from the process.
           In these cases, skip ncclCommDestroy */
          return;
        }
        comm_destroy(comms[i]);
      }
    }
  }
  ArrayRef<ncclComm_t> ref() const {
    return ArrayRef<ncclComm_t>(comms.get(), ndevices);
  }
};

using device_list = std::vector<int>;
// accesses to this object have to be guarded by THC's CudaFreeMutex
static std::unordered_map<device_list, NcclCommList, torch::hash<device_list>>
    _communicators;

ArrayRef<ncclComm_t> get_communicators(TensorList inputs) {
  static auto get_device = [](const at::Tensor& t) -> int {
    return t.get_device();
  };
  device_list devices = fmap(inputs, get_device);
  auto it = _communicators.find(devices);
  if (it == _communicators.end())
    std::tie(it, std::ignore) = _communicators.emplace(devices, devices);
  return it->second.ref();
}

torchNcclDataType_t get_data_type(const Tensor& t) {
  if (!t.is_cuda()) {
    throw std::runtime_error("Unconvertible NCCL type");
  }
  switch (t.scalar_type()) {
    case at::kFloat:
      return ncclFloat;
    case at::kHalf:
      return ncclHalf;
    case at::kDouble:
      return ncclDouble;
    case at::kLong:
      return ncclInt64;
    case at::kInt:
      return ncclInt;
    case at::kChar:
      return ncclChar;
    case at::kByte:
      return ncclChar;
    default:
      throw std::runtime_error("Unconvertible NCCL type");
  }
}

static inline
void check_tensor(
    const at::Tensor& input,
    const at::optional<at::Tensor>& output,
    int input_multiplier,
    int output_multiplier,
    int64_t ref_numel,
    ScalarType ref_dtype) {

  auto check_one = [&](const at::Tensor &tensor) {
    if (!tensor.is_cuda() || tensor.is_sparse()) {
      throw std::runtime_error(
          "input and output elements have to be cuda dense Tensors");
    }

    if (ref_dtype != tensor.scalar_type()) {
      throw std::runtime_error(
          "all inputs and outputs must be of the same Tensor dtype");
    }

    if (!tensor.is_contiguous()) {
      throw std::runtime_error("all inputs and outputs have to be contiguous");
    }
  };

  check_one(input);

  // all inputs must be same size
  if (input.numel() != ref_numel) {
    throw std::runtime_error(
        "all inputs must have the same number of elements");
  }

  if (output) {
    check_one(*output);

    // inputs and outputs must be on same device respectively
    if (input.get_device() != output->get_device()) {
      throw std::runtime_error("input and output must be on the same device");
    }

    if (output->numel() * output_multiplier != ref_numel * input_multiplier) {
      throw std::runtime_error(
          "output must be of size input_size * size_multiplier");
    }
  }
}

void check_inputs(
    TensorList inputs,
    TensorList outputs,
    int input_multiplier,
    int output_multiplier) {
  // len(inputs) == len(outputs)
  size_t len = inputs.size();

  if (len <= 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  if (len != outputs.size()) {
    std::stringstream err;
    err << "inputs and outputs sequences have to be of the same length, but got input of length "
        << len << " and output of length " << outputs.size();
    throw std::runtime_error(err.str());
  }

  device_set devices;
  int64_t numel = inputs[0].numel();
  auto dtype = inputs[0].scalar_type();

  for (size_t i = 0; i < len; i++) {
    auto input = inputs[i];
    auto output = outputs[i];

    check_tensor(input, output, input_multiplier, output_multiplier, numel, dtype);

    auto input_device = input.get_device();
    // inputs must be on unique devices
    if (devices.test(input_device)) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    devices.set(input_device);
  }
}

void check_inputs(
    TensorList inputs,
    const at::Tensor& output,
    int root,
    int input_multiplier,
    int output_multiplier) {
  size_t len = inputs.size();

  if (len <= 0) {
    throw std::runtime_error("input sequence can't be empty");
  }

  device_set devices;
  int64_t numel = inputs[0].numel();
  auto dtype = inputs[0].scalar_type();

  for (size_t i = 0; i < len; i++) {
    auto input = inputs[i];

    check_tensor(
      input,
      i == root ? at::optional<at::Tensor>{output} : at::nullopt,
      input_multiplier, output_multiplier, numel, dtype);

    auto input_device = input.get_device();
    // inputs must be on unique devices
    if (devices.test(input_device)) {
      throw std::runtime_error("inputs must be on unique devices");
    }
    devices.set(input_device);
  }
}

} // namespace detail

bool is_available(TensorList tensors) {
#ifdef USE_NCCL
  device_set devices;
  for (auto& tensor : tensors) {
    if (!tensor.is_cuda() || tensor.is_sparse())
      return false;
    if (!tensor.is_contiguous())
      return false;
    auto device = tensor.get_device();
    if (devices[device])
      return false;
    devices[device] = true;
  }
  return true;
#else
  return false;
#endif
}

std::uint64_t version() {
#if defined(NCCL_MAJOR)
  return NCCL_MAJOR * 1000 + NCCL_MINOR * 100 + NCCL_PATCH;
#elif defined(USE_NCCL)
  return 1000;
#else
  return 0;
#endif
}

void get_unique_id(torchNcclUniqueId& id) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  ncclUniqueId* nccl_id = reinterpret_cast<ncclUniqueId*>(&id);
  NCCL_CHECK(static_cast<torchNcclResult_t>(ncclGetUniqueId(nccl_id)));
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

ncclComm_t comm_init_rank(
    int nranks,
    const torchNcclUniqueId& comm_id,
    int rank) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  ncclComm_t comm;
  const ncclUniqueId* nccl_id = reinterpret_cast<const ncclUniqueId*>(&comm_id);
  NCCL_CHECK(
      static_cast<torchNcclResult_t>(ncclCommInitRank(&comm, nranks, *nccl_id, rank)));
  return comm;
#else
  return nullptr;
#endif
}

void comm_destroy(ncclComm_t comm)
{
  /*
   * TODO(T30279827) Temporarily disable calling ncclCommDestroy
   * Calling ncclCommDestroy while program exiting is undefined
   * according to Nvidia, and lead to segfault in NCCL 2
   * (whether it is called before or after the CUDA runtime destructor).
   * Temporarily disable it in destructor to avoid segfault.
   * Following up with Nvidia for long term solution.
   */
  return;

#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  NCCL_CHECK(static_cast<torchNcclResult_t>(ncclCommDestroy(comm)));
#endif
}

namespace {
// NCCL changed the numerical type used for count between NCCL1 and NCCL2.
// So we use the following struct, which gets the type of the second argument
// of T, if T is a function type, with ncclBcast, to get that type statically
// and programmatically.

template <typename T>
struct GetSecondArgType;

template <typename R, typename Arg0, typename Arg1, typename... Args>
struct GetSecondArgType<R(Arg0, Arg1, Args...)> {
  typedef typename std::decay<Arg1>::type type;
};

constexpr auto count_max =
    std::numeric_limits<GetSecondArgType<decltype(ncclBcast)>::type>::max();
} // namespace

size_t get_max_count() {
  return count_max;
}

void broadcast(
    TensorList tensors,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  check_inputs(tensors, tensors, 1, 1);
  ncclDataType_t data_type =
      static_cast < ncclDataType_t>(get_data_type(tensors[0]));
  int64_t numel = tensors[0].numel();

  const auto comms = user_comms.empty() ? get_communicators(tensors)
                                        : ArrayRef<ncclComm_t>(user_comms);

  AutoNcclGroup nccl_group_guard;
  at::cuda::OptionalCUDAGuard device_guard;
  for (size_t i = 0, num_tensors = tensors.size(); i < num_tensors; i++) {
    int device = tensors[i].get_device();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();
    TORCH_CHECK(
        static_cast<uint64_t>(numel) <= static_cast<uint64_t>(count_max),
        "Broadcast tensor has ",
        numel,
        " elements, which exceeds the "
        "maximum NCCL supports (",
        count_max,
        ")");
    NCCL_CHECK(static_cast<torchNcclResult_t>(ncclBcast(
        tensors[i].data_ptr(), numel, data_type, 0, comms[i], stream)));
  }
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void reduce(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& output,
    int32_t root,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  TORCH_CHECK(
      root >= 0 && static_cast<size_t>(root) < inputs.size(), "invalid root");

  check_inputs(inputs, output, root, 1, 1);
  const auto len = inputs.size();

  ncclDataType_t data_type =
      static_cast<ncclDataType_t>(get_data_type(inputs[0]));

  const auto count = inputs[0].numel();
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  AutoNcclGroup nccl_group_guard;
  at::cuda::OptionalCUDAGuard device_guard;
  for (size_t i = 0; i < len; i++) {
    int device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    NCCL_CHECK(
        static_cast<torchNcclResult_t>(ncclReduce(
        inputs[i].data_ptr(),
        root == i ? output.data_ptr() : nullptr,
        count,
        data_type,
        (ncclRedOp_t)op,
        root,
        comms_ref[i],
        stream)));
  }
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
  reduce(inputs, /*output=*/inputs[root], root, op, streams, user_comms);
}

void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  check_inputs(inputs, outputs, 1, 1);
  const auto len = inputs.size();

  ncclDataType_t data_type =
      static_cast<ncclDataType_t>(get_data_type(inputs[0]));

  const auto count = inputs[0].numel();
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  AutoNcclGroup nccl_group_guard;
  at::cuda::OptionalCUDAGuard device_guard;
  for (size_t i = 0; i < len; i++) {
    int device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    NCCL_CHECK(static_cast<torchNcclResult_t>(ncclAllReduce(
        inputs[i].data_ptr(),
        outputs[i].data_ptr(),
        count,
        data_type,
        (ncclRedOp_t)op,
        comms_ref[i],
        stream)));
  }
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  const auto len = inputs.size();
  check_inputs(inputs, outputs, 1, len);

  ncclDataType_t data_type =
      static_cast<ncclDataType_t>(get_data_type(inputs[0]));

  const auto count = inputs[0].numel() / len;
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  AutoNcclGroup nccl_group_guard;
  at::cuda::OptionalCUDAGuard device_guard;
  for (size_t i = 0; i < len; i++) {
    int device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

    NCCL_CHECK(static_cast<torchNcclResult_t>(ncclReduceScatter(
        inputs[i].data_ptr(),
        outputs[i].data_ptr(),
        count,
        data_type,
        (ncclRedOp_t)op,
        comms_ref[i],
        stream)));
  }
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}

void all_gather(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    const stream_list& streams,
    const comm_list& user_comms) {
#ifdef USE_NCCL
  using namespace torch::cuda::nccl::detail;
  const auto len = inputs.size();
  check_inputs(inputs, outputs, len, 1);

  ncclDataType_t data_type = static_cast<ncclDataType_t>(get_data_type(inputs[0]));

  const auto count = inputs[0].numel();
  auto comms_ref = user_comms.empty() ? get_communicators(inputs)
                                      : ArrayRef<ncclComm_t>(user_comms);

  AutoNcclGroup nccl_group_guard;
  at::cuda::OptionalCUDAGuard device_guard;
  for (size_t i = 0; i < len; i++) {
    int device = inputs[i].device().index();
    device_guard.set_index(device);
    // Default to the current stream
    const auto stream = (streams.empty() || !streams[i])
        ? at::cuda::getCurrentCUDAStream(device).stream()
        : streams[i]->stream();

#if defined(NCCL_MAJOR) && (NCCL_MAJOR >= 2)
      NCCL_CHECK(static_cast<torchNcclResult_t>(ncclAllGather(
          inputs[i].data_ptr(),
          outputs[i].data_ptr(),
          count,
          data_type,
          comms_ref[i],
          stream)));
#else
      NCCL_CHECK(static_cast<torchNcclResult_t>(ncclAllGather(
          inputs[i].data_ptr(),
          count,
          data_type,
          outputs[i].data_ptr(),
          comms_ref[i],
          stream)));
#endif
  }
#else
  AT_ERROR("PyTorch built without NCCL support");
#endif
}
} // namespace nccl
} // namespace cuda
} // namespace torch
