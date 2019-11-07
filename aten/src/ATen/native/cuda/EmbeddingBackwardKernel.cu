#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/TensorUtils.h>
#include <ATen/NativeFunctions.h>

#include <ATen/AccumulateType.h>

#include <THC/THCDeviceUtils.cuh>
#include <THC/THCTensorMathReduce.cuh>
#include <THC/THCTensorSort.cuh>
#include <THC/THCThrustAllocator.cuh>
#include <THC/THCAtomics.cuh>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>

#include <c10/macros/Macros.h>

namespace at {
namespace native {

namespace {

// The maximum block size in CUDA
constexpr int MAX_BLOCK_SIZE = 1024;
/* This code computes the sum of the weights in two-steps:
  1) Each GPU warp sums `NROWS_PER_THREAD` number of row given by `indeces`
  2) Each partial-sum from 1) are summed and scatter into `grad_weight`

  Notice, `NROWS_PER_THREAD` impacts the Achieved Occupancy of the
  kernel execution. If it is high, the size of the thread blocks will be
  too small to achieve good occupancy. Similarly, a very low value will
  make the size of the thread blocks in the final sum in step 2) too small.
*/
constexpr int NROWS_PER_THREAD = 10;

// Fast ceil division (no overflow checking)
__host__ __device__ __forceinline__
int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

__global__
void krn_partials_per_segment(int64_t *ret, const int64_t *segment_offsets,
                              int64_t num_of_segments, int64_t numel) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_of_segments) {
    const int64_t idx_start = segment_offsets[id];
    const int64_t idx_end = (id == num_of_segments-1)?numel:segment_offsets[id+1];
    const int64_t size = idx_end - idx_start;
    ret[id] = ceil_div(size, NROWS_PER_THREAD);
  }
}

__global__
void krn_partial_segment_offset(
        int64_t *ret,
        const int64_t *partials_per_segment,
        const int64_t *partials_per_segment_offset,
        const int64_t *segment_offsets,
        int64_t num_of_segments) {
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id < num_of_segments) {
    int64_t idx = partials_per_segment_offset[id];
    const int64_t num_partials = partials_per_segment[id];
    const int64_t segment_offset = segment_offsets[id];
    for (int64_t i=0; i<num_partials; ++i) {
      ret[idx++] = segment_offset + i * NROWS_PER_THREAD;
    }
  }
}


template <typename scalar_t>
__global__ void compute_grad_weight_bags(
    int64_t *indices, scalar_t *gradOutput,
    int64_t *offset2bag, int64_t *count, ptrdiff_t numel,
    int64_t stride, int mode_mean, const int64_t *bag_size,
    scalar_t* per_sample_weights, int64_t per_sample_weights_stride,
    int64_t* segment_offsets, int64_t num_of_segments,
    acc_type<scalar_t, true> *grad_weight_per_segment,
    const int64_t stride_warped) {

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = gid / stride_warped;
  const int startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }
  const int idx_begin = segment_offsets[id];
  const int idx_end = (id == num_of_segments-1)?numel:segment_offsets[id+1];

  acc_type<scalar_t, true> weight = 0;
  for (int idx=idx_begin; idx < idx_end; ++idx) {
    const int origRow = indices[idx];
    const int seq_number = offset2bag[origRow];
    const int gradOutputRow = seq_number * stride;

    acc_type<scalar_t, true> scale = count ? 1.0 / count[idx] : 1.0;
    if (per_sample_weights) {
      scale *= per_sample_weights[origRow * per_sample_weights_stride];
    }

    acc_type<scalar_t, true> gradient = gradOutput[gradOutputRow + startFeature];
    if (mode_mean) {
      gradient /= bag_size[seq_number];
    }
    weight += gradient * scale;
  }
  grad_weight_per_segment[id * stride + startFeature] = weight;
}

template <typename scalar_t>
__global__ void compute_grad_weight(
    int64_t *indices,
    scalar_t *gradOutput,
    int64_t *count,
    ptrdiff_t numel,
    int64_t stride,
    int64_t* segment_offsets,
    int64_t num_of_segments,
    acc_type<scalar_t, true> *grad_weight_per_segment,
    const int64_t stride_warped) {

  using accscalar_t = acc_type<scalar_t, true>;
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = gid / stride_warped;
  const int startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }
  const int idx_begin = segment_offsets[id];
  const int idx_end = (id == num_of_segments-1)?numel:segment_offsets[id+1];

  accscalar_t weight = 0;
  for (int idx=idx_begin; idx < idx_end; ++idx) {
    const int64_t target_row = indices[idx];
    const accscalar_t scale = count ? (accscalar_t)1.0 / count[idx] : 1.0;
    weight += gradOutput[target_row * stride + startFeature] * scale;
  }
  grad_weight_per_segment[id * stride + startFeature] = weight;
}

// This kernel assumes that all input tensors are contiguous.
template <typename scalar_t>
__global__ void sum_and_scatter(
    int64_t *input, scalar_t *gradWeight, int64_t stride,
    int64_t* segment_offsets, int64_t num_of_segments,
    const acc_type<scalar_t, true> *grad_weight_per_segment,
    const int64_t *segment_sizes_offsets, int64_t num_of_partial_segments,
    const int64_t padding_idx,
    const int64_t stride_warped) {

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int id = gid / stride_warped;
  const int startFeature = gid % stride_warped;
  if (startFeature >= stride) {
    return;
  }
  if (id >= num_of_segments) {
    return;
  }

  const int idx_begin = segment_sizes_offsets[id];
  const int idx_end = (id == num_of_segments-1)?num_of_partial_segments:segment_sizes_offsets[id+1];
  acc_type<scalar_t, true> weight = 0;
  for (int idx=idx_begin; idx < idx_end; ++idx) {
    weight += grad_weight_per_segment[idx*stride + startFeature];
  }
  int64_t target_row = input[segment_offsets[id]];
  if (target_row != padding_idx) {
    gradWeight[target_row * stride + startFeature] = weight;
  }
}

} // anon namespace

Tensor embedding_backward_cuda_kernel(
        const Tensor &grad,
        const Tensor &orig_indices,
        const Tensor &sorted_indices,
        const Tensor &count,
        int64_t num_weights,
        int padding_idx,
        bool scale_grad_by_freq,
        bool mode_mean,
        const Tensor &offset2bag,
        const Tensor &bag_size,
        const Tensor &per_sample_weights) {

  auto stream = at::cuda::getCurrentCUDAStream();
  auto allocator = THCThrustAllocator(globalContext().lazyInitCUDA());
  auto policy = thrust::cuda::par(allocator).on(stream);
  const ptrdiff_t numel = sorted_indices.numel();

  auto grad_weight = at::zeros({num_weights, grad.size(-1)}, grad.options());
  const int64_t stride = grad_weight.stride(0);

  // Compute the number of segments and their start position so that we do not have to
  // spawn a warp per index. In this context, a segment is a number of rows that should
  // be summarized.
  // Unit: index in `sorted_indices` and `orig_indices`
  auto segment_offsets = at::empty({numel}, orig_indices.options());
  int64_t num_of_segments;
  {
    auto sorted_indices_dev = thrust::device_ptr<int64_t>(sorted_indices.data_ptr<int64_t>());
    auto dummy = at::empty_like(sorted_indices, at::MemoryFormat::Contiguous);
    auto dummy_dev = thrust::device_ptr<int64_t>(dummy.data_ptr<int64_t>());
    auto ends = thrust::unique_by_key_copy(
            policy,
            sorted_indices_dev,
            sorted_indices_dev + numel,
            thrust::make_counting_iterator(0),
            dummy_dev,
            thrust::device_ptr<int64_t>(segment_offsets.data_ptr<int64_t>()));
    num_of_segments = thrust::get<0>(ends) - dummy_dev;
  }

  // We split the segments up into sizes of `NROWS_PER_THREAD`
  // Compute the number partial-segments per segment (some partial-segments 
  // may not be the full `NROWS_PER_THREAD` number of rows)
  auto partials_per_segment = at::empty({num_of_segments}, orig_indices.options());
  {
    krn_partials_per_segment<<<ceil_div(num_of_segments, 32), 32, 0, stream>>> (
            partials_per_segment.data_ptr<int64_t>(),
            segment_offsets.data_ptr<int64_t>(),
            num_of_segments,
            numel);
  }

  // In order to compute `partial_segment_offset`, which is the start index
  // of each partial-segment in `sorted_indices`, we need to compute the
  // start position of each _segment_ in `partial_segment_offset`.
  // Unit: index in `partial_segment_offset`
  auto partials_per_segment_offset = at::empty({num_of_segments}, orig_indices.options());
  thrust::exclusive_scan(
          policy,
          thrust::device_ptr<int64_t>(partials_per_segment.data_ptr<int64_t>()),
          thrust::device_ptr<int64_t>(partials_per_segment.data_ptr<int64_t>()+num_of_segments),
          thrust::device_ptr<int64_t>(partials_per_segment_offset.data_ptr<int64_t>()));

  // The total number of partial-segments is the sum of `partials_per_segment_offset`
  const int num_of_partial_segments = partials_per_segment[num_of_segments-1].item<int64_t>() +
          partials_per_segment_offset[num_of_segments-1].item<int64_t>();

  // Now we can compute the start position of each partial-segment
  // Unit: index in `sorted_indices` and `orig_indices`
  auto partial_segment_offset = at::empty({num_of_partial_segments}, orig_indices.options());
  {
    krn_partial_segment_offset<<<ceil_div(num_of_segments, 32), 32, 0, stream>>> (
            partial_segment_offset.data_ptr<int64_t>(),
            partials_per_segment.data_ptr<int64_t>(),
            partials_per_segment_offset.data_ptr<int64_t>(),
            segment_offsets.data_ptr<int64_t>(),
            num_of_segments);
  }

  const int stride_warped = ceil_div(stride, C10_WARP_SIZE)*C10_WARP_SIZE;
  const int block = std::min(stride_warped, MAX_BLOCK_SIZE);
  const int grid = ceil_div(num_of_partial_segments*stride_warped, block);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "embedding_bag_backward_cuda_compute_grad_weight", [&] {
      // For numerical stability, the dtype of `grad_weight_per_segment`
      // should match `acc_type`
      using partial_weight_t = acc_type<scalar_t, true>;
      TensorOptions op;
      if(grad.dtype() == at::kHalf) {
          op = grad.options().dtype(at::kFloat);
      } else {
          op = grad.options();
      }
      auto grad_weight_per_segment = at::empty({num_of_partial_segments, stride}, op);
      // Compute the sum of each partial-segment and handle bags
      if (offset2bag.defined()) {
            compute_grad_weight_bags<scalar_t><<<grid, block, 0, stream>>>(
              orig_indices.data_ptr<int64_t>(),
              grad.data_ptr<scalar_t>(),
              offset2bag.data_ptr<int64_t>(),
              count.defined() ? count.data_ptr<int64_t>() : nullptr, numel, stride,
              mode_mean, bag_size.data_ptr<int64_t>(),
              per_sample_weights.defined() ? per_sample_weights.data_ptr<scalar_t>() : NULL,
              per_sample_weights.defined() ? per_sample_weights.stride(0) : 0,
              partial_segment_offset.data_ptr<int64_t>(),
              num_of_partial_segments, grad_weight_per_segment.data_ptr<partial_weight_t>(),
              stride_warped);
      } else {
            compute_grad_weight<scalar_t><<<grid, block, 0, stream>>>(
              orig_indices.data_ptr<int64_t>(),
              grad.data_ptr<scalar_t>(),
              count.defined() ? count.data_ptr<int64_t>() : nullptr,
              numel, stride,
              partial_segment_offset.data_ptr<int64_t>(),
              num_of_partial_segments,
              grad_weight_per_segment.data_ptr<partial_weight_t>(),
              stride_warped);
      }
      THCudaCheck(cudaGetLastError());

      // Finally, we sum all the partial-sums and scatter them
      // into `grad_weight`.
      const int grid2 = ceil_div(num_of_segments*stride_warped, block);
          sum_and_scatter<scalar_t><<<grid2, block, 0, stream>>>(
            sorted_indices.data_ptr<int64_t>(),
            grad_weight.data_ptr<scalar_t>(),
            stride,
            segment_offsets.data_ptr<int64_t>(),
            num_of_segments, grad_weight_per_segment.data_ptr<partial_weight_t>(),
            partials_per_segment_offset.data_ptr<int64_t>(),
            num_of_partial_segments, 
            padding_idx, 
            stride_warped);
      THCudaCheck(cudaGetLastError());
  });
  return grad_weight;
}

}}
