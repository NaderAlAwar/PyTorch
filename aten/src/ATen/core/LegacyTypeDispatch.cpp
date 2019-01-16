#include <ATen/core/LegacyTypeDispatch.h>

namespace at {

/// NOTE [ Treating Variables as non-Variables in type dispatch ]
///
/// Previously, in VariableType_*.cpp (generated by gen_variable_type.py), when
/// a function is using the 'use_derived' strategy, we call its implementation
/// on the base non-Variable type (`baseType`), passing unwrapped tensors to the
/// call so that any `.type()` calls in the implementation can treat the passed
/// tensors as non-Variables and won't dispatch back to functions in VariableType.
///
/// However, after the Variable/Tensor merge, there is no concept of unwrapping
/// a tensor anymore, and directly passing variables to the base type calls will
/// cause the `.type()` dispatch in the implementation to treat the tensor as a
/// variable, and any function dispatch based on `.type()` will dispatch back to
/// VariableType, which is not what we want.
///
/// The solution to the above problem is to add `at::NonVariableTypeMode`, which
/// when enabled will cause `legacyTensorType()` and `getType()` to always return
/// non-Variable type, even if the tensor being called on is a variable.

/// In the CAFFE2_FB_LIMITED_MOBILE_CAPABILITY build setting,
/// thread_local is not supported. In that case, we don't provide
/// `at::NonVariableTypeMode`.
#if !C10_MOBILE && !defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

thread_local bool NonVariableTypeMode_enabled = false;

bool NonVariableTypeMode::is_enabled() {
  return NonVariableTypeMode_enabled;
}

void NonVariableTypeMode::set_enabled(bool enabled) {
  NonVariableTypeMode_enabled = enabled;
}

#else // C10_MOBILE || defined(CAFFE2_FB_LIMITED_MOBILE_CAPABILITY)

bool NonVariableTypeMode::is_enabled() {
  throw std::runtime_error("NonVariableTypeMode is not supported on mobile");
}

void NonVariableTypeMode::set_enabled(bool enabled) {
  throw std::runtime_error("NonVariableTypeMode is not supported on mobile");
}

#endif

// TODO: This could be bad juju if someone calls globalContext() in the
// destructor of an object with static lifetime.
LegacyTypeDispatch & globalLegacyTypeDispatch() {
  static LegacyTypeDispatch singleton;
  return singleton;
}

}
