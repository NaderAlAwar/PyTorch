#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/native/cpu/ReducedPrecisionFloatGemvFastPathKernel.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/Half.h>
#include <c10/util/Unroll.h>

#if defined(__aarch64__) && !defined(C10_MOBILE)
#include <arm_neon.h>
#include <cpuinfo.h>
#endif

namespace at::native {
inline namespace CPU_CAPABILITY {
#if defined(__aarch64__) && !defined(C10_MOBILE)

// We need the shift for reduce(), hence the extra constants.
constexpr auto kF16ElementsPerIterationShift = 7;
constexpr auto kF16ElementsPerIteration = 1 << kF16ElementsPerIterationShift;
static_assert(kF16ElementsPerIteration == 128);

constexpr auto kF16ElementsPerRegisterShift = 3;
constexpr auto kF16ElementsPerRegister = 1 << kF16ElementsPerRegisterShift;
static_assert(kF16ElementsPerRegister == 8);

constexpr auto kF16RegistersPerIterationShift = kF16ElementsPerIterationShift - kF16ElementsPerRegisterShift;
constexpr auto kF16RegistersPerIteration = 1 << kF16RegistersPerIterationShift;
static_assert(kF16RegistersPerIteration == kF16ElementsPerIteration / kF16ElementsPerRegister);

constexpr auto kF32ElementsPerIterationShift = 5;
constexpr auto kF32ElementsPerIteration = 1 << kF32ElementsPerIterationShift;
static_assert(kF32ElementsPerIteration == 32);

constexpr auto kF32ElementsPerRegisterShift = 2;
constexpr auto kF32ElementsPerRegister = 1 << kF32ElementsPerRegisterShift;
static_assert(kF32ElementsPerRegister == 4);

constexpr auto kF32RegisterPairsPerIteration = 4;
constexpr auto kF32RegistersPerIteration = kF32RegisterPairsPerIteration * 2;
constexpr auto kF32RegistersPerIterationShift = 3;
static_assert(kF32RegistersPerIteration == kF32ElementsPerIteration / kF32ElementsPerRegister);
static_assert(kF32RegistersPerIteration == 1 << kF32RegistersPerIterationShift);

/*
 * NOTE [ GGML Copyright Notice ]
 * The below reduce overload and fp16_dot_with_fp16_arith function is
 * adapted from llama.cpp's ggml_vec_dot_f16 and surrounding utility
 * functions, so here is the required copyright notice:
 *
 * MIT License
 *
 * Copyright (c) 2023-2024 The ggml authors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
static inline float reduce(vec::VectorizedN<Half, kF16RegistersPerIteration>& x) {
  int offset = kF16RegistersPerIteration;
  c10::ForcedUnroll<kF16RegistersPerIterationShift>{}([&offset, &x](auto idx) {
    offset /= 2;
    for (int i = 0; i < offset; ++i) {
      x[i] = x[i] + x[offset + i];
    }
  });
  const auto [t0, t1] = vec::convert_half_float(x[0]);
  return vaddvq_f32(t0 + t1);
}

float fp16_dot_with_fp16_arith(const Half* x, const Half* a, int len) {
  vec::VectorizedN<Half, kF16RegistersPerIteration> sum(0);

  const auto len_aligned = len & ~(kF16ElementsPerIteration - 1);
  for (int j = 0; j < len_aligned ; j += kF16ElementsPerIteration) {
    for (int k = 0; k < kF16RegistersPerIteration; ++k) {
      const auto temp_x = vec::Vectorized<Half>::loadu(x + j + k * vec::Vectorized<Half>::size());
      const auto temp_a = vec::Vectorized<Half>::loadu(a + j + k * vec::Vectorized<Half>::size());
      sum[k] = vec::fmadd(temp_x, temp_a, sum[k]);
    }
  }
  auto reduced_sum = reduce(sum);

  for (int j = len_aligned; j < len; ++j) {
    reduced_sum += x[j] * a[j];
  }
  return reduced_sum;
}

// Rather than unrolling to process multiple rows (transposed columns)
// of matrix A at once as done in fp16_gemv_trans_fp16_arith, unroll
// along an individual dot product.
static void fp16_gemv_trans_fp16_arith_by_dot_products(const int m, const int n, const Half* a, const int lda, const Half *x, Half* y, int incy) {
  parallel_for(0, n, 1, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      y[i * incy] = fp16_dot_with_fp16_arith(x, a + lda * i, m);
    }
  });
}

#endif // __ARM_FEATURE_FP16_SCALAR_ARITHMETIC

// The below reduce overload and fp16_dot_with_fp32_arith are adapted
// from llama.cpp's ggml_vec_dot_f32 and surrounding utility
// functions. See NOTE [ GGML Copyright Notice ] above for the
// required notice.
static inline float reduce(vec::VectorizedN<float, kF32RegistersPerIteration>& x) {
  int offset = kF32RegistersPerIteration;
  c10::ForcedUnroll<kF32RegistersPerIterationShift>{}([&offset, &x](auto idx) {
    offset /= 2;
    for (int i = 0; i < offset; ++i) {
      x[i] = vaddq_f32(x[i], x[offset + i]);
    }
  });
  return vaddvq_f32(x[0]);
}

namespace {
// Returns (acc_low + a_low_half * b_low_half, acc_high + a_high_half * b_high_half)
std::pair<vec::Vectorized<float>, vec::Vectorized<float>> fmadd(
    const vec::Vectorized<c10::Half>& a,
    const vec::Vectorized<c10::Half>& b,
    const vec::Vectorized<float>& acc_low,
    const vec::Vectorized<float>& acc_high) {
#ifdef __ARM_FEATURE_FP16_FML
  return std::make_pair(vfmlalq_low_f16(acc_low, a, b), vfmlalq_high_f16(acc_high, a, b));
#else
  const auto [a_float_low, a_float_high] = convert_half_float(a);
  const auto [b_float_low, b_float_high] = convert_half_float(b);
  return std::make_pair(fmadd(a_float_low, b_float_low, acc_low), fmadd(a_float_high, b_float_high, acc_high));
#endif
}
} // namespace

C10_ALWAYS_INLINE void dot_with_fp32_arith_main_inner_loop_no_bfdot(
  const Half* vec1,
  const Half* vec2,
  vec::VectorizedN<float, kF32RegistersPerIteration>& sum,
  int registerPairIndex) {
  // Load a pair of f32 registers at a time.
  const auto temp_vec1 = vec::Vectorized<Half>::loadu(&vec1[registerPairIndex * vec::Vectorized<Half>::size()]);
  const auto temp_vec2 = vec::Vectorized<Half>::loadu(&vec2[registerPairIndex * vec::Vectorized<Half>::size()]);

  const auto [result_low, result_high] = fmadd(temp_vec1, temp_vec2, sum[2 * registerPairIndex], sum[2 * registerPairIndex + 1]);
  sum[2 * registerPairIndex] = result_low;
  sum[2 * registerPairIndex + 1] = result_high;
}

inline float32x4_t f32_fma_f16(float32x4_t a, float16x4_t b, float16x4_t c) {
#ifdef __ARM_FEATURE_FP16_FML
  // NOTE: this instruction is an optional instruction in ARM v8.2 and
  // v8.3, but mandatory in v8.4 per
  // https://developer.arm.com/documentation/ddi0596/2021-03/SIMD-FP-Instructions/FMLAL--FMLAL2--vector---Floating-point-fused-Multiply-Add-Long-to-accumulator--vector--?lang=en
  // I'm not certain that I have the right feature test macro.
  return vfmlalq_low_f16(a, vcombine_f16(b, vdup_n_f16(0)), vcombine_f16(c, vdup_n_f16(0)));
#else
  return vec::fmadd(vec::Vectorized<float>(vcvt_f32_f16(b)), vec::Vectorized<float>(vcvt_f32_f16(c)), vec::Vectorized<float>(a));
#endif
}

C10_ALWAYS_INLINE void dot_with_fp32_arith_vectorized_tail_inner_loop_no_bfdot(
    const Half* vec1,
    const Half* vec2,
    vec::Vectorized<float>* tail_sum,
    int idx) {
  const auto temp_vec1 = vld1_f16(reinterpret_cast<const float16_t*>(&vec1[idx]));
  const auto temp_vec2 = vld1_f16(reinterpret_cast<const float16_t*>(&vec2[idx]));
  *tail_sum = f32_fma_f16(*tail_sum, temp_vec1, temp_vec2);
}

// NOTE: This and some below code is unfortunately duplicated with
// BlasKernel.cpp for now, but will be consolidated when BF16 moves
// here as well.
template <typename T>
C10_ALWAYS_INLINE auto
dot_with_fp32_arith_main_loop_no_bfdot(
    const T* vec1,
    const T* vec2,
    int64_t len) {
  vec::VectorizedN<float, kF32RegistersPerIteration> sum(0);
  const auto len_aligned = len & ~(kF32ElementsPerIteration - 1);
  for (int j = 0; j < len_aligned ; j += kF32ElementsPerIteration) {
    const auto* vec1_ = vec1 + j;
    const auto* vec2_ = vec2 + j;
    c10::ForcedUnroll<kF32RegisterPairsPerIteration>{}([vec1_, vec2_, &sum](auto k) C10_ALWAYS_INLINE_ATTRIBUTE {
      dot_with_fp32_arith_main_inner_loop_no_bfdot(vec1_, vec2_, sum, k);
    });
  }
  return reduce(sum);
}

template <typename T>
struct half_to_float16 {
  using type = T;
};


#ifdef __aarch64__
template <>
struct half_to_float16<Half> {
  using type = float16_t;
};
#endif

template <typename T>
using half_to_float16_t = typename half_to_float16<T>::type;

// NOTE [GCC code duplication]: The first attempt at landing BFDOT support with
// TARGET_ARM_BF16_ATTRIBUTE failed because unlike clang, GCC will not
// allow inlining a non-bf16-specific function into a bf16-specific
// function. We can work around this by duplicating the code into the
// bfdot and non-bfdot callsites. The code is in this macro to avoid
// actual copy/paste.
#define DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY(bfdot_suffix)     \
  /* First-tier tail fixup: make sure we handle workloads that can */   \
  /* benefit from vectorization, but don't fit into our fully unrolled */ \
  /* loop above. */                                                     \
  vec::Vectorized<float> tail_sum(0);                                   \
  const auto len_aligned = len & ~(kF32ElementsPerIteration - 1);       \
  const auto len_aligned_4 = len & ~3;                                  \
  for (int j = len_aligned; j < len_aligned_4; j += 4) {                \
    dot_with_fp32_arith_vectorized_tail_inner_loop##bfdot_suffix(vec1, vec2, &tail_sum, j); \
  }                                                                     \
  auto reduced_tail = vpaddq_f32(tail_sum, tail_sum);                   \
  reduced_sum += vgetq_lane_f32(vpaddq_f32(reduced_tail, reduced_tail), 0); \
                                                                        \
  /* Second-tier tail fixup: handle all workloads. */                   \
  for (int j = len_aligned_4; j < len; ++j) {                           \
    /* We use half_to_float16_t here because changing to Half was */    \
    /* causing arithmetic to at fp16 precision, but the necessary */    \
    /* necessary behavior to pass python test/test_mps.py -k */         \
    /* test_output_grad_match_nn_functional_linear_cpu_float16 is */    \
    /* fp32. (I'm not sure exactly why this fixes it.) */               \
    half_to_float16_t<std::decay_t<decltype(vec1[j])>> x1 = vec1[j];    \
    half_to_float16_t<std::decay_t<decltype(vec2[j])>> x2 = vec2[j];    \
    reduced_sum += x1 * x2;                                             \
  }                                                                     \
  return reduced_sum

template <typename T>
C10_ALWAYS_INLINE float
dot_with_fp32_arith_no_bfdot(const T* vec1, const T* vec2, int64_t len) {
  auto reduced_sum = dot_with_fp32_arith_main_loop_no_bfdot(vec1, vec2, len);
  DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY(_no_bfdot);
}
#undef DOT_WITH_FP32_ARITH_TAIL_AFTER_MAIN_LOOP_BODY

float fp16_dot_with_fp32_arith(const Half* vec1, const Half* vec2, int64_t len) {
  return dot_with_fp32_arith_no_bfdot(vec1, vec2, len);
}

// On my Apple M1 Macbook (which is ARM v8.5 and thus has the
// instructions f32_fma_{low,high}_f16 is targeting), this kernel has
// equivalent performance to the fp16-native kernel.
void fp16_gemv_trans_fp32_arith_by_dot_products(const int m, const int n, const Half* a, const int lda, const Half *x, Half* y, int incy) {
  parallel_for(0, n, 1, [&](int begin, int end) {
    for (int i = begin; i < end; ++i) {
      y[i * incy] = fp16_dot_with_fp32_arith(x, a + lda * i, m);
    }
  });
}

void fp16_gemv_trans(
    const int m,
    const int n,
    const float alpha,
    const Half* a,
    const int lda,
    const Half* x,
    const int incx,
    const float beta,
    Half* y,
    const int incy) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(incx == 1 && alpha == 1.0 && beta == 0.0);
#ifdef __ARM_FEATURE_FP16_SCALAR_ARITHMETIC
  if (at::globalContext().allowFP16ReductionCPU()) {
    return fp16_gemv_trans_fp16_arith_by_dot_products(m, n, a, lda, x, y, incy);
  }
#endif
  return fp16_gemv_trans_fp32_arith_by_dot_products(m, n, a, lda, x, y, incy);
}
#endif // defined(__aarch64__) && !defined(C10_MOBILE)

} // namespace CPU_CAPABILITY

#if defined(__aarch64__) && !defined(C10_MOBILE)
// NOTE: we don't *need* to go through dispatch for the ARM-only
// implementation right now, but we will need it when we cover x86.
REGISTER_DISPATCH(fp16_dot_with_fp32_arith_stub, &fp16_dot_with_fp32_arith);
REGISTER_DISPATCH(fp16_gemv_trans_stub, &fp16_gemv_trans);
#else
#endif // defined(__aarch64__) && !defined(C10_MOBILE)

} // namespace at::native
