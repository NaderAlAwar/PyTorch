#pragma once

#include <ATen/ATen.h>
#include <stdexcept>
#include "CapabilityDispatch.h"

namespace at {
namespace native {

using unary_fn = void (*)(Tensor&, const Tensor&);

extern DispatchStub<unary_fn> absImpl;
extern DispatchStub<unary_fn> acosImpl;
extern DispatchStub<unary_fn> asinImpl;
extern DispatchStub<unary_fn> atanImpl;
extern DispatchStub<unary_fn> ceilImpl;
extern DispatchStub<unary_fn> erfImpl;
extern DispatchStub<unary_fn> expImpl;
extern DispatchStub<unary_fn> expm1Impl;
extern DispatchStub<unary_fn> fracImpl;
extern DispatchStub<unary_fn> floorImpl;
extern DispatchStub<unary_fn> logImpl;
extern DispatchStub<unary_fn> log10Impl;
extern DispatchStub<unary_fn> log1pImpl;
extern DispatchStub<unary_fn> log2Impl;
extern DispatchStub<unary_fn> roundImpl;
extern DispatchStub<unary_fn> rsqrtImpl;
extern DispatchStub<unary_fn> sqrtImpl;
extern DispatchStub<unary_fn> tanhImpl;
extern DispatchStub<unary_fn> truncImpl;

extern DispatchStub<unary_fn> cosImpl;
extern DispatchStub<unary_fn> coshImpl;
extern DispatchStub<unary_fn> sinImpl;
extern DispatchStub<unary_fn> sinhImpl;
extern DispatchStub<unary_fn> tanImpl;

extern DispatchStub<void (*)(Tensor&, Scalar&)> fillImpl;
extern DispatchStub<void (*)(Tensor&, const Tensor&, Scalar&, Scalar&)>
    clampImpl;
extern DispatchStub<void (*)(Tensor&, const Tensor&, Scalar&)> clampMinImpl;
extern DispatchStub<void (*)(Tensor&, const Tensor&, Scalar&)> clampMaxImpl;

// Missing unary functions
// digamma
// lgamma

} // namespace native
} // namespace at
