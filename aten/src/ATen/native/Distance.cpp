#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Distance.h>

namespace at { namespace native {

DEFINE_DISPATCH(pdist_forward_stub);
DEFINE_DISPATCH(pdist_backward_stub);
DEFINE_DISPATCH(cdist_stub);
DEFINE_DISPATCH(cdist_backward_stub);

Tensor pairwise_distance(const Tensor& x1, const Tensor& x2, double p, double eps, bool keepdim) {
  return at::norm(x1 - x2 + eps, p, 1, keepdim);
}

// This is to guarantee that the contiguous memory is passed to the backward pass
Tensor pdist(const Tensor& self, const double p) {
  TORCH_CHECK(self.dim() == 2,
      "pdist only supports 2D tensors, got: ", self.dim(), "D");
  TORCH_CHECK(at::isFloatingType(self.scalar_type()), "pdist only supports floating-point dtypes");
  TORCH_CHECK(p >= 0, "pdist only supports non-negative p values");
  return at::_pdist_forward(self.contiguous(), p);
}

Tensor cdist(const Tensor& x1, const Tensor& x2, const double p) {
  TORCH_CHECK(x1.dim() >= 2, "cdist only supports at least 2D tensors, X1 got: ", x1.dim(), "D");
  TORCH_CHECK(at::isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X1 got: ", x1.scalar_type());
  auto device1 = x1.type().device_type();
  TORCH_CHECK(device1 == kCPU || device1 == kCUDA, "cdist only supports CPU and CUDA devices, X1 got: ", device1);
  TORCH_CHECK(x2.dim() >= 2, "cdist only supports at least 2D tensors, X2 got: ", x2.dim(), "D");
  TORCH_CHECK(at::isFloatingType(x1.scalar_type()), "cdist only supports floating-point dtypes, X2 got: ", x2.scalar_type());
  auto device2 = x2.type().device_type();
  TORCH_CHECK(device2 == kCPU || device2 == kCUDA, "cdist only supports CPU and CUDA devices, X2 got: ", device2);
  TORCH_CHECK(p >= 0, "cdist only supports non-negative p values");
  TORCH_CHECK(device1 == device2, "X1 and X2 must have the same device type. X1: ", device1, " X2: ", device2);
  TORCH_CHECK(!x1.is_cuda() || x1.get_device() == x2.get_device(), "device of X1 (", x1.get_device(), ") must match device of X2 (", x2.get_device(), ")");
  int64_t c1 = x1.size(-1);
  int64_t c2 = x2.size(-1);
  TORCH_CHECK(c1 == c2, "X1 and X2 must have the same number of columns. X1: ", c1, " X2: ", c2);

  int64_t r1 = x1.size(-2);
  int64_t r2 = x2.size(-2);
  auto dim1 = x1.dim();
  auto dim2 = x2.dim();

  IntArrayRef batch_tensor1(x1.sizes().data(), std::max<int64_t>(dim1 - 2, 0));
  IntArrayRef batch_tensor2(x2.sizes().data(), std::max<int64_t>(dim2 - 2, 0));
  std::vector<int64_t> expand_batch_portion = infer_size(batch_tensor1, batch_tensor2);
  std::vector<int64_t> tensor1_expand_size(expand_batch_portion);
  tensor1_expand_size.insert(tensor1_expand_size.end(), {r1, c1});
  std::vector<int64_t> tensor2_expand_size(expand_batch_portion);
  tensor2_expand_size.insert(tensor2_expand_size.end(), {r2, c2});

  int expand_batch_product = std::accumulate(expand_batch_portion.begin(), expand_batch_portion.end(), 1, std::multiplies<int64_t>());
  std::vector<int64_t> tensor1_view({expand_batch_product});
  tensor1_view.insert(tensor1_view.end(), {r1, c1});
  std::vector<int64_t> tensor2_view({expand_batch_product});
  tensor2_view.insert(tensor2_view.end(), {r2, c2});

  Tensor tensor1_expanded = x1.expand(tensor1_expand_size).contiguous().view(tensor1_view);
  Tensor tensor2_expanded = x2.expand(tensor2_expand_size).contiguous().view(tensor2_view);

  std::vector<int64_t> output_shape(expand_batch_portion);
  output_shape.push_back(r1);
  output_shape.push_back(r2);

  Tensor result = at::empty(output_shape, x1.options());
  if (r1 > 0 && r2 > 0) {
    if (c1 == 0) {
      result.fill_(0);
    } else {
      cdist_stub(device1, result, tensor1_expanded, tensor2_expanded, p);
    }
  }
  return result;
}

Tensor _cdist_backward(const Tensor& grad, const Tensor& x1, const Tensor& x2, const double p, const Tensor& cdist) {
  TORCH_CHECK(x1.is_contiguous(), "_cdist_backward requires X1 to be contiguous");
  TORCH_CHECK(x2.is_contiguous(), "_cdist_backward requires X2 to be contiguous");
  TORCH_CHECK(cdist.is_contiguous(), "_cdist_backward requires dist to be contiguous");
  int64_t n = x1.size(-2);
  int64_t m = x1.size(-1);
  auto device1 = x1.type().device_type();
  TORCH_CHECK(device1 == kCPU || device1 == kCUDA, "_cdist_backward only supports CPU and CUDA devices, X1 got: ", device1);
  auto device2 = x2.type().device_type();
  TORCH_CHECK(device2 == kCPU || device2 == kCUDA, "_cdist_backward only supports CPU and CUDA devices, X2 got: ", device2);
  IntArrayRef batch_tensor1(x1.sizes().data(), std::max<int64_t>(x1.dim() - 2, 0));
  int batch_product = std::accumulate(batch_tensor1.begin(), batch_tensor1.end(), 1, std::multiplies<int64_t>());
  Tensor grad_x1 = at::empty_like(x1, x1.options()).view({batch_product, n, m});
  cdist_backward_stub(device1, grad_x1, grad, x1, x2, p, cdist);
  return grad_x1;
}

Tensor _pdist_forward(const Tensor& self, const double p) {
  TORCH_CHECK(self.is_contiguous(), "_pdist_forward requires contiguous input");
  auto device = self.type().device_type();
  TORCH_CHECK(device == kCPU || device == kCUDA, "_pdist_forward only supports CPU and CUDA devices, got: ", device);
  Tensor result = at::empty({0}, self.options());
  if (self.size(0) <= 1) {
    result.resize_({0});
  } else {
    int64_t n = self.size(0);
    int64_t c = n * (n - 1) / 2;
    result.resize_({c});
    if (self.size(1) == 0) {
      result.fill_(0);
    } else {
      pdist_forward_stub(device, result, self, p);
    }
  }
  return result;
}

Tensor _pdist_backward(const Tensor& grad, const Tensor& self, const double p, const Tensor& pdist) {
  TORCH_CHECK(self.is_contiguous(), "_pdist_backward requires self to be contiguous");
  TORCH_CHECK(pdist.is_contiguous(), "_pdist_backward requires pdist to be contiguous");
  auto device = self.type().device_type();
  TORCH_CHECK(device == kCPU || device == kCUDA, "_pdist_backward only supports CPU and CUDA devices, got: ", device);
  Tensor result = at::empty_like(self);
  pdist_backward_stub(device, result, grad, self, p, pdist);
  return result;
}

Tensor cosine_similarity(const Tensor& x1, const Tensor& x2, int64_t dim, double eps) {
  // Follow scipy impl to improve numerical precision
  // Use x / sqrt(x * x) instead of x / (sqrt(x) * sqrt(x))
  Tensor w12 = at::sum(x1 * x2, dim);
  Tensor w1 = at::sum(x1 * x1, dim);
  Tensor w2 = at::sum(x2 * x2, dim);
  Tensor n12 = (w1 * w2).clamp_min_(eps * eps).sqrt_();
  return w12.div_(n12);
}

}}  // namespace at::native
