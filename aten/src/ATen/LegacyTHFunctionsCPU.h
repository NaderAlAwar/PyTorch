#pragma once

#include <ATen/Context.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

namespace c10 {
class Scalar;
}
namespace at {
struct Generator;
class Tensor;
struct Type;
} // namespace at

namespace at {
namespace native {
namespace legacy {
namespace cpu {

Tensor & _th_masked_scatter_(Tensor & self, const Tensor & mask, const Tensor & source);
Tensor & _th_masked_scatter_bool_(Tensor & self, const Tensor & mask, const Tensor & source);
Tensor & _th_histc_out(const Tensor & self, int64_t bins, const Scalar& min, const Scalar& max, Tensor & result);
Tensor _th_histc(const Tensor & self, int64_t bins, const Scalar& min, const Scalar& max);
std::tuple<Tensor &,Tensor &> _th_gels_out(const Tensor & self, const Tensor & A, Tensor & res1, Tensor & res2);
std::tuple<Tensor,Tensor> _th_gels(const Tensor & self, const Tensor & A);

} // namespace th
} // namespace legacy
} // namespace native
} // namespace at
