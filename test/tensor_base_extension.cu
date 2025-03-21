#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void tensor_base_add_kernel(float* __restrict__ x,
                                      float* __restrict__ y,
                                      float* __restrict__ out,
                                      const int size) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = x[idx] + y[idx] + 1.0f; // Add 1 to make it distinguishable
  }
}

at::TensorBase tensor_base_add(const at::TensorBase& x, const at::TensorBase& y) {
  // Validate inputs
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, "x must be a float tensor");
  TORCH_CHECK(y.scalar_type() == at::ScalarType::Float, "y must be a float tensor");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

  // Create output tensor with same properties as input
  auto options = at::TensorOptions()
                    .dtype(x.scalar_type())
                    .device(x.device())
                    .layout(x.layout());
  
  at::TensorBase output = at::empty_like(x, options);
  
  const int num_elements = x.numel();
  const int threads = 1024;
  const int blocks = (num_elements + threads - 1) / threads;
  
  // Get raw pointers
  float* x_data = x.data_ptr<float>();
  float* y_data = y.data_ptr<float>();
  float* output_data = output.data_ptr<float>();

  // Set CUDA device and launch kernel
  const at::cuda::CUDAGuard device_guard(x.device());
  tensor_base_add_kernel<<<blocks, threads>>>(
      x_data, y_data, output_data, num_elements);
      
  return output;
}