#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include <gloo/algorithm.h>
#include <gloo/common/error.h>
#include <gloo/context.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/device.h>

#include <torch/csrc/utils/hash.h>

#ifdef USE_CUDA
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/cuda/CUDAStream.h>
#endif

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>

namespace c10d {

// ProcessGroupGloo implements Gloo bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group. This is the only way that we
// can guarantee to match up the same calls across processes. For
// multi-threaded usage of process groups, you can use consider using
// multiple process group instances.
//
// The Gloo algorithms that this class calls into are cached by their
// signature (see description of AlgorithmKey above). This cache works
// as follows: every function call instantiates an AlgorithmKey and
// looks in the cache for existing entries. If there is one, it is
// removed from the cache and returned to the caller. If there are
// none, a new entry is created and returned. If an entry was created
// before, but is still in use, the call will block and wait until the
// entry is returned to the cache.
//
// In the future, we hope to extend this to allow multiple entries per
// key, to enable parallelism for a single key. The number of entries
// per key must always be identical for all processes. This maximum
// number can be automatically tuned, but only if we let a single
// process take charge, and have it broadcast the limits.
//
class ProcessGroupGloo : public ProcessGroup {
 public:
  // AsyncWork is the Gloo specific superclass for asynchronous work items.
  // We can split asynchronous work into 3 phases:
  // 1) Sanity checks and prepare input (e.g. memcpy)
  // 2) Run operation on background thread
  // 3) Synchronize with completion on foreground thread
  //
  // There is state to be shared between these 3 phases and all of this state
  // is captured in the AsyncWork class and its derivatives.
  //
  // Note: while we are porting operations to use new style collectives, there
  // is a split between operations using the existing caching approach and
  // operations using the new AsyncWork base class. Over time we will port
  // all operations and perform needed cleanup.
  //
  class AsyncWork : public ProcessGroup::Work {
   public:
    bool isCompleted() override;
    bool isSuccess() const override;
    void synchronize() override;
    bool wait() override;
    const std::exception& exception() const override;

    static void execute(std::shared_ptr<AsyncWork> work) {
      std::exception_ptr eptr;
      try {
        work->run();
      } catch (...) {
        eptr = std::current_exception();
      }
      work->finish(eptr);
    }

    virtual void run() = 0;

   protected:
    std::mutex m_;
    std::condition_variable cv_;
    bool completed_ = false;
    std::exception_ptr eptr_;

    void finish(std::exception_ptr ptr);

    friend class ProcessGroupGloo;
  };

  // For send and recv operations there is no need to pass them to the
  // thread pool as they are entirely completed by the device thread.
  // This work object is used to synchronize completion of the send or
  // recv operation. It keeps a reference to the tensor it is
  // operating on to prevent it from being deallocated while the
  // operation is still in flight.
  class SendWork : public ProcessGroup::Work {
   public:
    explicit SendWork(
        at::Tensor& tensor,
        std::unique_ptr<::gloo::transport::UnboundBuffer> buffer);

    virtual ~SendWork() = default;

    bool isCompleted() override;

    bool isSuccess() const override;

    void synchronize() override;

    bool wait() override;

    const std::exception& exception() const override;

   protected:
    at::Tensor tensor_;
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
  };

  class RecvWork : public ProcessGroup::Work {
   public:
    explicit RecvWork(
        at::Tensor& tensor,
        std::unique_ptr<::gloo::transport::UnboundBuffer> buffer,
        int* srcRank);

    virtual ~RecvWork() = default;

    bool isCompleted() override;

    bool isSuccess() const override;

    void synchronize() override;

    bool wait() override;

    const std::exception& exception() const override;

   protected:
    at::Tensor tensor_;
    std::unique_ptr<::gloo::transport::UnboundBuffer> buffer_;
    int* srcRank_;
  };

  struct Options {
    explicit Options();

    std::vector<std::shared_ptr<::gloo::transport::Device>> devices;
    std::chrono::milliseconds timeout;
    int threads;

    // This controls how many Gloo algorithm instances are created for
    // a single identifying key. If you have many identical calls with
    // tensors of identical size and need to parallelize, this should
    // be greater than 1. More cache entries means more memory usage.
    // The default value is 1.
    int cacheNumAlgorithmEntries;
  };

  explicit ProcessGroupGloo(
      const std::shared_ptr<Store>& store,
      int rank,
      int size,
      Options options = Options());

  virtual ~ProcessGroupGloo();

  std::shared_ptr<Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs) override;

  std::shared_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputs,
      std::vector<at::Tensor>& inputs,
      const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<Work> scatter(
      std::vector<at::Tensor>& outputs,
      std::vector<std::vector<at::Tensor>>& inputs,
      const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  std::shared_ptr<Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  std::shared_ptr<Work> recvAnysource(
      std::vector<at::Tensor>& tensors,
      int* srcRank,
      int tag) override;

  std::shared_ptr<Work> barrier() override;

  std::unordered_map<int, int> getGroupRank() override;

 protected:
  std::unique_ptr<::gloo::rendezvous::Store> store_;
  std::vector<std::shared_ptr<::gloo::Context>> contexts_;
  std::vector<std::thread> threads_;
  bool stop_;

  // Incremented for every collective we kick off.
  // The value is used as tag for collective operations. Collectives are kicked
  // off in identical order across processes. Therefore the tag can be used
  // to match up operations during concurrent execution.
  uint32_t collectiveCounter_;

  // Returns next collective tag to use (uses collectiveCounter_).
  uint32_t nextTag();

  // Entrypoint for worker threads.
  void runLoop(void);

  // Queue std::function to run on worker thread.
  void enqueue(std::function<void()> fn);

  std::deque<std::function<void()>> queue_;
  std::mutex queueMutex_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;
};

} // namespace c10d
