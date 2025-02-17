#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/TensorIterator.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/mps/OperationUtils.h>

#include <fmt/format.h>

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/UnaryKernel_metallib.h>
#endif

static void exec_unary_kernel(const Tensor& self,
                              const Tensor& output_,
                              const std::string& name,
                              std::optional<int64_t> extra = std::nullopt) {
  Tensor inputTensor = self.contiguous();
  Tensor outputTensor = output_;
  bool needs_output_copy = false;
  uint32_t length = output_.numel();
  if (length == 0) {
    return;
  }
  using namespace mps;
  @autoreleasepool {
    id<MTLComputePipelineState> cplState = nil;
    if (c10::isComplexType(self.scalar_type())) {
      auto scalarStr = self.scalar_type() == kComplexFloat ? "float" : "half";
      cplState = lib.getPipelineStateForFunc(fmt::format("{}_complex_{}_{}", name, scalarStr, scalarStr));
    } else {
      cplState = lib.getPipelineStateForFunc(
          fmt::format("{}_{}_{}", name, scalarToMetalTypeString(outputTensor), scalarToMetalTypeString(self)));
    }

    if (!outputTensor.is_contiguous()) {
      outputTensor = outputTensor.contiguous();
      needs_output_copy = true;
    }

    MPSStream* mpsStream = getCurrentMPSStream();
    dispatch_sync(mpsStream->queue(), ^() {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      getMPSProfiler().beginProfileKernel(cplState, name, {self});

      [computeEncoder setComputePipelineState:cplState];
      mtl_setArgs(computeEncoder, outputTensor, inputTensor);
      if (extra) {
        mtl_setBytes(computeEncoder, *extra, 2);
      }
      mtl_dispatch1DJob(computeEncoder, cplState, length);

      getMPSProfiler().endProfileKernel(cplState);
    });
  }
  if (needs_output_copy) {
    output_.copy_(outputTensor);
  }
}

static void erfinv_kernel(TensorIteratorBase& iter) {
  exec_unary_kernel(iter.input(0), iter.output(0), "erfinv");
}

static void exp_kernel(TensorIteratorBase& iter) {
  exec_unary_kernel(iter.input(0), iter.output(0), "exp");
}

static void sinc_kernel(TensorIteratorBase& iter) {
  exec_unary_kernel(iter.input(0), iter.output(0), "sinc");
}

static void tanh_kernel(TensorIteratorBase& iter) {
  exec_unary_kernel(iter.input(0), iter.output(0), "tanh");
}

static void round_decimals_kernel(TensorIteratorBase& iter, int64_t decimals) {
  exec_unary_kernel(iter.input(0), iter.output(0), "round_decimals", decimals);
}

REGISTER_DISPATCH(exp_stub, exp_kernel);
REGISTER_DISPATCH(erfinv_stub, erfinv_kernel);
REGISTER_DISPATCH(sinc_stub, sinc_kernel);
REGISTER_DISPATCH(tanh_stub, tanh_kernel);
REGISTER_DISPATCH(round_decimals_stub, round_decimals_kernel);
} // namespace at::native
