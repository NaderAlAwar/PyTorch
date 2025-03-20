#pragma once

// NOTE: [pytorch nccl defines]

// All NCCL interactions should route through this header.
// Direct inclusion of <nccl.h> should be avoided.
// Version checks/compatibility macros centralized here.

#include <nccl.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstddef>
#include <optional>
#include <vector>

static_assert(
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 7, 0),
    "NCCL version must be 2.7 or later");

// NCCL BFloat16 is enabled only for CUDA 11+ and NCCL versions 2.10+, or for
// HIP 3.1+
#if defined(__CUDA_BF16_TYPES_EXIST__) && \
    (NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0))
#define NCCL_HAS_BF16_DATATYPE
#elif defined(RCCL_BFLOAT16)
#define NCCL_HAS_BF16_DATATYPE
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
#define NCCL_HAS_AVG
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 11, 0)
#define ENABLE_NCCL_PREMUL_SUM_SUPPORT
#endif

// ncclGetLastError() is enabled only for NCCL versions 2.13+
// ncclRemoteError only exists in NCCL versions 2.13+
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 13, 0)
#define NCCL_HAS_REMOTE_ERROR
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 14, 0)
#define NCCL_HAS_COMM_NONBLOCKING
#endif

// Note: the first version that supports ncclConfig_t is 2.14. Here we
// fast-forward the version requirement to 2.17 where ncclConfig_t has CTA and
// CGA fields because they have already been pybinded out.
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 17, 0)
#define NCCL_HAS_CONFIG
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 18, 0)
#define NCCL_HAS_COMM_SPLIT
#endif

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 19, 0)
#define NCCL_HAS_COMM_REGISTER
#define NCCL_HAS_MEM_ALLOC
#endif

namespace torch::cuda::nccl {

/* The following are copied from <nccl.h> and redefined in torch::cuda::nccl
 * namespace */
/* pytorch should only use the following definition within pytorch scope */

/* Opaque handle to communicator to ncclComm*, this will reinterpret as ncclComm
 * in nccl.cpp */
typedef void* ncclComm_t;

/** redefine nccl unique ID in torch scope. this should be identical to native
 * nccl impp. */
#define NCCL_UNIQUE_ID_BYTES 128
typedef struct {
  // NOLINTNEXTLINE(*array*)
  char internal[NCCL_UNIQUE_ID_BYTES];
} ncclUniqueId;

/* Error type */
enum class ncclResult {
  Success = 0,
  UnhandledCudaError = 1,
  SystemError = 2,
  InternalError = 3,
  InvalidArgument = 4,
  InvalidUsage = 5,
  RemoteError = 6,
  InProgress = 7,
  NumResults = 8
};

/* Reduction operation selector */
enum class ncclRedOp { Sum = 0, Prod = 1, Max = 2, Min = 3, NumOps = 4 };

/* Data types */
enum class ncclDataType {
  Int8 = 0,
  Char = 0,
  Uint8 = 1,
  Int32 = 2,
  Int = 2,
  Uint32 = 3,
  Int64 = 4,
  Uint64 = 5,
  Float16 = 6,
  Half = 6,
  Float32 = 7,
  Float = 7,
  Float64 = 8,
  Double = 8,
  Bfloat16 = 9,
  NumTypes = 10
};

// RAII helper class to manage NCCL group API and CUDA free mutex.
// The destructor is allowed to throw since this helper class only
// manages group and lock lifetimes.
struct TORCH_CUDA_CPP_API AutoNcclGroup {
  AutoNcclGroup();
  AutoNcclGroup(ncclComm_t comm, bool comm_nonblocking);
  ~AutoNcclGroup() noexcept(false);
  ncclComm_t comm_;
  bool comm_nonblocking_;
};

// NOTE: this is exposed only so that python_nccl.cpp can some of these helpers.
// Don't use them outside of these files.
namespace detail {

TORCH_CUDA_CPP_API void throw_nccl_error(ncclResult status);

inline void NCCL_CHECK(ncclResult status) {
  if (status != ncclResult::Success) {
    throw_nccl_error(status);
  }
}

TORCH_CUDA_CPP_API at::ArrayRef<ncclComm_t> get_communicators(
    at::TensorList inputs);
TORCH_CUDA_CPP_API void check_inputs(
    at::TensorList inputs,
    at::TensorList outputs,
    size_t input_multiplier,
    size_t output_multiplier);
TORCH_CUDA_CPP_API void check_inputs(
    at::TensorList inputs,
    const at::Tensor& output,
    int root,
    size_t input_multiplier,
    size_t output_multiplier);

} // namespace detail

using comm_list = std::vector<ncclComm_t>;
using stream_list = std::vector<std::optional<at::cuda::CUDAStream>>;

TORCH_CUDA_CPP_API std::uint64_t version();
TORCH_CUDA_CPP_API const char* version_suffix();

bool is_available(at::TensorList tensors);

TORCH_CUDA_CPP_API void get_unique_id(ncclUniqueId& id);
TORCH_CUDA_CPP_API ncclComm_t
comm_init_rank(int nranks, const ncclUniqueId& comm_id, int rank);
TORCH_CUDA_CPP_API void comm_destroy(ncclComm_t comm);

TORCH_CUDA_CPP_API void broadcast(
    at::TensorList tensors,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

size_t get_max_count();

TORCH_CUDA_CPP_API void reduce(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& output,
    int32_t root = 0,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_CPP_API void reduce(
    std::vector<at::Tensor>& inputs,
    int32_t root = 0,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_CPP_API void all_reduce(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_CPP_API void reduce_scatter(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    int32_t op = static_cast<int>(ncclRedOp::Sum),
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_CPP_API void scatter(
    const std::vector<at::Tensor>& inputs,
    at::Tensor& outputs,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream,
    int32_t root = 0);

TORCH_CUDA_CPP_API void all_gather(
    const std::vector<at::Tensor>& inputs,
    std::vector<at::Tensor>& outputs,
    const stream_list& streams = {},
    const comm_list& user_comms = {});

TORCH_CUDA_CPP_API void gather(
    const at::Tensor& inputs,
    std::vector<at::Tensor>& outputs,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream,
    int32_t root = 0);

TORCH_CUDA_CPP_API void all2all_single_equal_split(
    at::Tensor& input,
    at::Tensor& output,
    int size,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream);

TORCH_CUDA_CPP_API void all2all_single_unequal_split(
    void* sendbuff,
    const size_t* sendcounts,
    const size_t* senddispls,
    void* recvbuff,
    const size_t* recvcounts,
    const size_t* recvdispls,
    size_t size,
    c10::ScalarType type,
    ncclComm_t comm,
    at::cuda::CUDAStream& stream);

TORCH_CUDA_CPP_API void all2all(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    ncclComm_t _comm,
    at::cuda::CUDAStream& stream);

TORCH_CUDA_CPP_API void send(
    const at::Tensor& input,
    ncclComm_t comm,
    at::cuda::CUDAStream stream,
    int dst);

TORCH_CUDA_CPP_API void recv(
    at::Tensor& output,
    ncclComm_t comm,
    at::cuda::CUDAStream stream,
    int src);
} // namespace torch::cuda::nccl
