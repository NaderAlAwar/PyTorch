#include <cmath>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec256/vec256.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/Pow.h>
#include <ATen/native/cpu/Loops.h>

namespace at { namespace native {

namespace {

void pow_tensor_tensor_kernel(TensorIterator& iter) {
  if (isFloatingType(iter.dtype())) {
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "pow", [&]() {
      using Vec = Vec256<scalar_t>;
      cpu_kernel_vec(iter,
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return std::pow(base, exp);
        },
        [&](Vec base, Vec exp) -> Vec {
          return base.pow(exp);
        }
      );
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow", [&]() {
      cpu_kernel(iter,
        [=](scalar_t base, scalar_t exp) -> scalar_t {
          return std::pow(base, exp);
        }
      );
    });
  }
}

void pow_tensor_scalar_kernel(TensorIterator& iter, Scalar exp_scalar) {
  // Casting exponent to double(not tensor.dtype) allows powering integral
  // tensors to float exponent e.g. tensor([4]).pow(0.5) will be tensor([2])
  const auto exp = exp_scalar.to<double>();
  if (isFloatingType(iter.dtype())) {
    // Floating types allow AVX2 vector optimizations for pow/sqrt/rsqrt:
    AT_DISPATCH_FLOATING_TYPES(iter.dtype(), "pow", [&]() {
      using Vec = Vec256<scalar_t>;
      if (exp == 0.5) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t {
            return std::sqrt((long double)base);
          },
          [](Vec base) -> Vec { return base.sqrt(); }
        );
      } else if (exp == 2) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t { return base * base; },
          [](Vec base) -> Vec { return base * base; }
        );
      } else if (exp == 3) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t { return base * base * base; },
          [](Vec base) -> Vec { return base * base * base; }
        );
      } else if (exp == -0.5) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t {
            return 1.0 / std::sqrt((long double)base);
          },
          [](Vec base) -> Vec { return base.rsqrt(); }
        );
      } else if (exp == -1) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t { return 1.0 / base; },
          [](Vec base) -> Vec { return base.reciprocal(); }
        );
      } else if (exp == -2) {
        cpu_kernel_vec(iter,
          [](scalar_t base) -> scalar_t { return 1.0 / (base * base); },
          [](Vec base) -> Vec { return (base * base).reciprocal(); }
        );
      } else {
        cpu_kernel_vec(iter,
          [=](scalar_t base) -> scalar_t {
            return std::pow((long double)base, exp);
          },
          [=](Vec base) -> Vec { return base.pow(exp); }
        );
      }
    });
  } else {
    // Integral types do not allow AVX2 vector optimizations for pow/sqrt/rsqrt.
    // Trying to implement pow/sqrt/rsqrt as loops in vec256_int.h does not allow
    // powering integral tensor to float exponent. That's why we need this code
    // duplication:
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "pow", [&]() {
      if (exp == 0.5) {
        cpu_kernel(iter,
          [](scalar_t base) -> scalar_t { return std::sqrt(base); }
        );
      } else if (exp == 2) {
        cpu_kernel(iter,
          [](scalar_t base) -> scalar_t { return base * base; }
        );
      } else if (exp == 3) {
        cpu_kernel(iter,
          [](scalar_t base) -> scalar_t { return base * base * base; }
        );
      } else if (exp == -0.5) {
        cpu_kernel(iter,
          [](scalar_t base) -> scalar_t { return 1.0 / std::sqrt(base); }
        );
      } else if (exp == -1) {
        cpu_kernel(iter,
          [](scalar_t base) -> scalar_t { return 1.0 / base; }
        );
      } else if (exp == -2) {
        cpu_kernel(iter,
          [](scalar_t base) -> scalar_t { return 1.0 / (base * base); }
        );
      } else {
        cpu_kernel(iter,
          [=](scalar_t base) -> scalar_t {
            return std::pow((long double)base, exp);
          }
        );
      }
    });
  }
}

void pow_scalar_tensor_kernel(TensorIterator& iter, Scalar base_scalar) {
  const long double base = base_scalar.isFloatingPoint() ?
                           static_cast<long double>(base_scalar.to<double>()) :
                           static_cast<long double>(base_scalar.to<int64_t>());
  AT_DISPATCH_ALL_TYPES(iter.dtype(), "pow", [&]() {
    cpu_kernel(iter,
      [=](scalar_t exp) -> scalar_t {
        return std::pow(base, exp);
      }
    );
  });
}

} // anonymous namespace

REGISTER_DISPATCH(pow_tensor_tensor_stub, &pow_tensor_tensor_kernel);
REGISTER_DISPATCH(pow_tensor_scalar_stub, &pow_tensor_scalar_kernel);
REGISTER_DISPATCH(pow_scalar_tensor_stub, &pow_scalar_tensor_kernel);

}} // namespace at::native
