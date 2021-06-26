// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <functorch/csrc/VmapTransforms.h>
#include <functorch/csrc/BatchedTensorImpl.h>
#include <functorch/csrc/PlumbingHelper.h>
#include <functorch/csrc/Constants.h>
#include <functorch/csrc/DynamicLayer.h>


namespace at {
namespace functorch {
template <typename... Args> Tensor unsupportedRandomOp(Args... args) {
  TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
              "Please perform random operations outside of vmap as a workaround");
}

template <typename... Args> Tensor& unsupportedRandomOp_(Args... args) {
  TORCH_CHECK(false, "vmap: We do not yet support calling random operations inside of vmap. ",
              "Please perform random operations outside of vmap as a workaround");
}

TORCH_LIBRARY_IMPL(_, FuncTorchVmapMode, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

#define TENSOROPTIONSPARAMS c10::optional<c10::ScalarType> dtype, c10::optional<c10::Layout> layout, c10::optional<c10::Device> device, c10::optional<bool> pin_memory

#define TENSOROPTIONSARGS dtype, layout, device, pin_memory

Tensor randn_mbatching_rule(IntArrayRef shape, TENSOROPTIONSPARAMS) {
    TORCH_WARN("Automatically expanding the shape of randn. This means that different elements of vmap will get different random values. If you want different behavior, like using the same random values across vmap, please make a github issue.");
    c10::impl::ExcludeDispatchKeyGuard guard(kVmapModeKey);
    auto maybe_layer = maybeCurrentDynamicLayer();
    VmapDimVector shapeVec(shape.begin(), shape.end());
    shapeVec.insert(shapeVec.begin(), maybe_layer->batchSize());
    return makeBatched(at::randn(shapeVec, TENSOROPTIONSARGS), 0, maybe_layer->layerId());
}

#undef TENSOROPTIONSARGS
#undef TENSOROPTIONSPARAMS

TORCH_LIBRARY_IMPL(aten, FuncTorchVmapMode, m) {
  // NB: I'd really like to register a special kernel like
  // CppFunction::makeNamedNotSupported() to avoid listing out the types of everything.
  // However, registering e.g. CppFunction::makeNamedNotSupported() as an implementation
  // only works for operators that support boxing.
#define TENSOROPTIONS c10::optional<c10::ScalarType>, c10::optional<c10::Layout>, c10::optional<c10::Device>, c10::optional<bool>

  // random operations (out-of-place)
  m.impl("bernoulli", unsupportedRandomOp<const Tensor&, optional<Generator>>);
  m.impl("bernoulli.out", unsupportedRandomOp_<const Tensor&, optional<Generator>, Tensor&>);
  m.impl("bernoulli.p", unsupportedRandomOp<const Tensor&, double, optional<Generator>>);
  m.impl("bernoulli_.Tensor", unsupportedRandomOp_<Tensor&, const Tensor&, optional<Generator>>);
  m.impl("bernoulli_.float", unsupportedRandomOp_<Tensor&, double, optional<Generator>>);

  m.impl("cauchy_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);
  m.impl("exponential_", unsupportedRandomOp_<Tensor&, double, optional<Generator>>);
  m.impl("geometric_", unsupportedRandomOp_<Tensor&, double, optional<Generator>>);
  m.impl("log_normal_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);
  m.impl("multinomial", unsupportedRandomOp<const Tensor&, int64_t, bool, optional<Generator>>);
  m.impl("multinomial.out", unsupportedRandomOp_<const Tensor&, int64_t, bool, optional<Generator>, Tensor&>);

  m.impl("normal.Tensor_float", unsupportedRandomOp<const Tensor&, double, optional<Generator>>);
  m.impl("normal.Tensor_float_out", unsupportedRandomOp_<const Tensor&, double, optional<Generator>, Tensor&>);
  m.impl("normal.float_Tensor_out", unsupportedRandomOp_<double, const Tensor&, optional<Generator>, Tensor&>);
  m.impl("normal.float_Tensor", unsupportedRandomOp<double, const Tensor&, optional<Generator>>);
  m.impl("normal.Tensor_Tensor", unsupportedRandomOp<const Tensor&, const Tensor&, optional<Generator>>);
  m.impl("normal.Tensor_Tensor_out", unsupportedRandomOp_<const Tensor&, const Tensor&, optional<Generator>, Tensor&>);
  m.impl("normal.float_float", unsupportedRandomOp<double, double, IntArrayRef, optional<Generator>, TENSOROPTIONS>);
  m.impl("normal.float_float_out", unsupportedRandomOp_<double, double, IntArrayRef, optional<Generator>, Tensor&>);
  m.impl("normal_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);

  m.impl("poisson", unsupportedRandomOp<const Tensor&, optional<Generator>>);

  m.impl("random_.from", unsupportedRandomOp_<Tensor&, int64_t, optional<int64_t>, optional<Generator>>);
  m.impl("random_.to", unsupportedRandomOp_<Tensor&, int64_t, optional<Generator>>);
  m.impl("random_", unsupportedRandomOp_<Tensor&, optional<Generator>>);

  // m.impl("rand_like", unsupportedRandomOp<const Tensor&, TENSOROPTIONS, optional<MemoryFormat>>);
  // m.impl("randn_like", unsupportedRandomOp<const Tensor&, TENSOROPTIONS, optional<MemoryFormat>>);

  m.impl("randint_like", unsupportedRandomOp<const Tensor&, int64_t, TENSOROPTIONS, optional<MemoryFormat>>);
  m.impl("randint_like.low_dtype", unsupportedRandomOp<const Tensor&, int64_t, int64_t, TENSOROPTIONS, optional<MemoryFormat>>);

  m.impl("rand", unsupportedRandomOp<IntArrayRef, TENSOROPTIONS>);
  m.impl("rand.generator", unsupportedRandomOp<IntArrayRef, optional<Generator>, TENSOROPTIONS>);
  m.impl("rand.names", unsupportedRandomOp<IntArrayRef, optional<DimnameList>, TENSOROPTIONS>);
  m.impl("rand.generator_with_names", unsupportedRandomOp<IntArrayRef, optional<Generator>, optional<DimnameList>, TENSOROPTIONS>);
  m.impl("rand.out", unsupportedRandomOp_<IntArrayRef, Tensor&>);
  m.impl("rand.generator_out", unsupportedRandomOp_<IntArrayRef, optional<Generator>, Tensor&>);

//   m.impl("randn", unsupportedRandomOp<IntArrayRef, TENSOROPTIONS>);
  m.impl("randn.generator", unsupportedRandomOp<IntArrayRef, optional<Generator>, TENSOROPTIONS>);
  m.impl("randn.names", unsupportedRandomOp<IntArrayRef, optional<DimnameList>, TENSOROPTIONS>);
  m.impl("randn.generator_with_names", unsupportedRandomOp<IntArrayRef, optional<Generator>, optional<DimnameList>, TENSOROPTIONS>);
  m.impl("randn.out", unsupportedRandomOp_<IntArrayRef, Tensor&>);
  m.impl("randn.generator_out", unsupportedRandomOp_<IntArrayRef, optional<Generator>, Tensor&>);

  m.impl("randperm", unsupportedRandomOp<int64_t, TENSOROPTIONS>);
  m.impl("randperm.generator", unsupportedRandomOp<int64_t, optional<Generator>, TENSOROPTIONS>);
  m.impl("randperm.out", unsupportedRandomOp_<int64_t, Tensor&>);
  m.impl("randperm.generator_out", unsupportedRandomOp_<int64_t, optional<Generator>, Tensor&>);

  m.impl("randint", unsupportedRandomOp<int64_t, IntArrayRef, TENSOROPTIONS>);
  m.impl("randint.generator", unsupportedRandomOp<int64_t, IntArrayRef, optional<Generator>, TENSOROPTIONS>);
  m.impl("randint.low", unsupportedRandomOp<int64_t, int64_t, IntArrayRef, TENSOROPTIONS>);
  m.impl("randint.low_generator", unsupportedRandomOp<int64_t, int64_t, IntArrayRef, optional<Generator>, TENSOROPTIONS>);
  m.impl("randint.out", unsupportedRandomOp_<int64_t, IntArrayRef, Tensor&>);
  m.impl("randint.generator_out", unsupportedRandomOp_<int64_t, IntArrayRef, optional<Generator>, Tensor&>);
  m.impl("randint.low_out", unsupportedRandomOp_<int64_t, int64_t, IntArrayRef, Tensor&>);
  m.impl("randint.low_generator_out", unsupportedRandomOp_<int64_t, int64_t, IntArrayRef, optional<Generator>, Tensor&>);

  m.impl("uniform_", unsupportedRandomOp_<Tensor&, double, double, optional<Generator>>);

#undef TENSOROPTIONS

  m.impl("randn", randn_mbatching_rule);
}


}
} // namespace at
