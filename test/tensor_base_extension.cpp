#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>

// Forward declaration of the CUDA kernel function
at::TensorBase tensor_base_add(const at::TensorBase& x, const at::TensorBase& y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tensor_base_add", &tensor_base_add, "Add two tensors using TensorBase directly");
}