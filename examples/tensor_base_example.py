"""
Example showing how to use the no_header mode with a TensorBase CUDA extension

This example creates a CUDA extension that directly includes ATen/core/TensorBase.h 
instead of torch/extension.h, resulting in faster compilation with no_header=True
"""
import torch
import torch.utils.cpp_extension

# C++ code that directly includes TensorBase.h without using torch/extension.h
cpp_source = """
#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAGuard.h>
#include <pybind11/pybind11.h>

// Forward declaration of the CUDA kernel function
at::TensorBase tensor_base_add(const at::TensorBase& x, const at::TensorBase& y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tensor_base_add", &tensor_base_add, "Add two tensors using TensorBase directly");
}
"""

# CUDA source with direct TensorBase usage
cuda_source = """
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
"""

def main():
    print("Compiling TensorBase CUDA extension with no_header=True...")
    # Load the extension using load_inline with no_header=True
    module = torch.utils.cpp_extension.load_inline(
        name="tensor_base_example",
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        verbose=True,
        no_header=True,  # Skip including torch/extension.h
    )
    
    print("Extension compiled successfully!")
    
    # Test the functionality
    print("Testing on CUDA tensors...")
    x = torch.randn(100, device="cuda", dtype=torch.float32)
    y = torch.randn(100, device="cuda", dtype=torch.float32)
    
    # Call our custom kernel
    result = module.tensor_base_add(x, y)
    
    # Verify result (our kernel adds 1.0 to distinguish it from a regular add)
    expected = x + y + 1.0
    
    # Check if results match
    if torch.allclose(result, expected):
        print("Test PASSED! ✓")
        print(f"First few elements of tensors:")
        print(f"x: {x[:5]}")
        print(f"y: {y[:5]}")
        print(f"result: {result[:5]}")
        print(f"expected: {expected[:5]}")
    else:
        print("Test FAILED!")
        max_diff = torch.max(torch.abs(result - expected))
        print(f"Maximum difference: {max_diff}")

if __name__ == "__main__":
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available, this example requires CUDA")
    else:
        main()