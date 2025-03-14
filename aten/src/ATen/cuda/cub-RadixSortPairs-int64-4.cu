#include <ATen/cuda/cub-RadixSortPairs.cuh>

namespace at::cuda::cub::detail {

AT_INSTANTIATE_SORT_PAIRS(int64_t, 4)

} // namespace at::cuda::cub::detail
