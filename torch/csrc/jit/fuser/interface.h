#pragma once

#include "ATen/ATen.h"
#include "torch/csrc/WindowsTorchApiMacro.h"
#include "torch/csrc/jit/ir.h"
#include "torch/csrc/jit/stack.h"

#include <memory>
#include <vector>
#include <cstdint>

namespace torch { namespace jit {

constexpr int kCPUDevice = -1;

struct TORCH_API FusionHandle {
  virtual void run(Stack& inputs) = 0;

  virtual ~FusionHandle() = 0;
};

TORCH_API void registerFusion(int64_t& key, const Node* fusion_group);
TORCH_API void runFusion(const int64_t key, Stack& stack);  

TORCH_API bool canFuseOnCPU();
TORCH_API bool canFuseOnGPU();

// CPU fuser is disabled by default, but we still want to test it.
TORCH_API void overrideCanFuseOnCPU(bool value);

TORCH_API std::vector<at::Tensor> debugLaunchGraph(
  Graph& graph
, int device
, at::ArrayRef<at::Tensor> inputs);

} // namespace jit
} // namespace torch
