#pragma once

#include <ATen/core/TensorBase.h>
#include <ATen/native/DispatchStub.h>

namespace at {
struct TensorIterator;
struct TensorIteratorBase;
}

namespace at::native {

using structured_beta_fn = void(*)(TensorIteratorBase&);

DECLARE_DISPATCH(structured_beta_fn, betainc_stub)

} // namespace at::native
