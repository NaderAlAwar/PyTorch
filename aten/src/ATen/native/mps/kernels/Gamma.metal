/*
 * The gamma function approximations follow John D Cook's
 * c++ implementation:  https://www.johndcook.com/Gamma.cpp.
 * (BSD License)
 *
 *
 * The digamma kernel and helper function is derived from the pytorch cpu
 * of this function, which is itself derived from the implementation
 * of the digamma function in the Cephes Math Library.
 * See note [3-Clause BSD License for the Cephes Math Library].
 */

#include <metal_stdlib>
using namespace metal;

template <typename T>
float LogGamma(const T);

template <typename T>
float Gamma(const T x) {
  if (x < 0.001) {
    constexpr float EULER_MASCHERONI = 0.577215664901532860606512090;
    // For small x, 1/Gamma(x) has power series x + gamma x^2  - ...
    // So in this range, 1/Gamma(x) = x + gamma x^2 with error on the order of
    // x^3. The relative error over this interval is less than 6e-7.

    return 1.0 / (x * (1.0 + EULER_MASCHERONI * x));
  }
  if (x >= 12.0) {
    return exp(LogGamma(x));
  }
  // The algorithm directly approximates gamma over (1,2) and uses
  // reduction identities to reduce other arguments to this interval.
  // numerator coefficients for gamma approximation over the interval (1,2)
  const float GAMMA_NUMERATOR_COEF[8] = {
      -1.71618513886549492533811E+0,
      2.47656508055759199108314E+1,
      -3.79804256470945635097577E+2,
      6.29331155312818442661052E+2,
      8.66966202790413211295064E+2,
      -3.14512729688483675254357E+4,
      -3.61444134186911729807069E+4,
      6.64561438202405440627855E+4};

  // denominator coefficients for gamma approximation over the interval (1,2)
  const float GAMMA_DENOMINATOR_COEF[8] = {
      -3.08402300119738975254353E+1,
      3.15350626979604161529144E+2,
      -1.01515636749021914166146E+3,
      -3.10777167157231109440444E+3,
      2.25381184209801510330112E+4,
      4.75584627752788110767815E+3,
      -1.34659959864969306392456E+5,
      -1.15132259675553483497211E+5};

  // Add or subtract integers as necessary to bring y into (1,2)
  float y = 1.0 + fract(x);

  float num = 0.0;
  float den = 1.0;

  float z = y - 1;
  for (int i = 0; i < 8; i++) {
    num = (num + GAMMA_NUMERATOR_COEF[i]) * z;
    den = den * z + GAMMA_DENOMINATOR_COEF[i];
  }
  float result = num / den + 1.0;

  // Apply correction if argument was not initially in (1,2)
  if (x < 1.0) {
    // identity gamma(z) = gamma(z+1)/z
    result /= (y - 1.0);
  } else {
    // identity gamma(z+n) = z*(z+1)* ... *(z+n-1)*gamma(z)
    auto n = static_cast<int>(floor(x));
    for (int i = 1; i < n; i++) {
      result *= y++;
    }
  }

  return result;
}

template <typename T>
float LogGamma(const T x) {
  constexpr float LOG_PI = 1.14472988584940017414342735135305;
  constexpr float HALF_LOG_TWO_PI = 0.91893853320467274178032973640562;
  constexpr float LGAMMA_EXPANSION_COEF[8] = {
      1.0 / 12.0,
      -1.0 / 360.0,
      1.0 / 1260.0,
      -1.0 / 1680.0,
      1.0 / 1188.0,
      -691.0 / 360360.0,
      1.0 / 156.0,
      -3617.0 / 122400.0};

  float logGamma;

  const auto abs_x = metal::abs(static_cast<float>(x));
  if (abs_x == 0) {
    return INFINITY;
  }
  if (abs_x < 12.0) {
    logGamma = log(fabs(Gamma(abs_x)));
  } else {
    // Abramowitz and Stegun 6.1.41
    // Asymptotic series should be good to at least 11 or 12 figures
    // For error analysis, see Whittiker and Watson
    // A Course in Modern Analysis (1927), page 252

    float z = 1.0 / (abs_x * abs_x);
    float sum = LGAMMA_EXPANSION_COEF[7];

    for (int i = 6; i >= 0; i--) {
      sum *= z;
      sum += LGAMMA_EXPANSION_COEF[i];
    }
    float series = sum / abs_x;

    logGamma = (abs_x - 0.5) * log(abs_x) - abs_x + HALF_LOG_TWO_PI + series;
  }

  if (x >= 0) {
    return logGamma;
  }

  return LOG_PI - logGamma -
      log(fabs(abs_x * sinpi(abs_x))); // Reflection Formula
}

float calc_digamma_positive_domain(float x) {
  const float DIGAMMA_COEF[7] = {
      8.33333333333333333333E-2,
      -2.10927960927960927961E-2,
      7.57575757575757575758E-3,
      -4.16666666666666666667E-3,
      3.96825396825396825397E-3,
      -8.33333333333333333333E-3,
      8.33333333333333333333E-2,
  };

  // Push x to be >= 10
  float result = 0;
  while (x < 10) {
    result -= 1 / x;
    x += 1;
  }
  if (x == 10) {
    constexpr float PSI_10 = 2.25175258906672110764;
    return result + PSI_10;
  }

  // Compute asymptotic digamma
  float y = 0;
  if (x < 1.0E+17) {
    float z = 1.0 / (x * x);
    for (int i = 0; i <= 6; i++) {
      y += pow(z, i) * DIGAMMA_COEF[i];
    }
    y *= z;
  }
  return result + log(x) - (0.5 / x) - y;
}

float calc_trigamma(float x) {
  float sign = 1.0f;
  float result = 0.0f;

  if (x < 0.0f) {
    sign = -1.0f;
    auto sin_pi_x = sin(M_PI_F * x);
    result -= (M_PI_F * M_PI_F) / (sin_pi_x * sin_pi_x);
    x = 1.0f - x;
  }

  else if (x == 0.0) {
    return INFINITY;
  }

  else if (x < 1.0) {
    result += 1.0 / (x * x);
    x += 1.0f;
  }

  for (int i = 0; i < 6; ++i) {
    result += 1.0f / (x * x);
    x += 1.0f;
  }

  const float ixx = 1.0f / (x * x);
  result +=
      (1.0f + 1.0f / (2.0f * x) +
       ixx * ((1.0f / 6.0f) - ixx * ((1.0f / 30.0f) - ixx * (1.0f / 42.0f)))) /
      x;
  return sign * result;
}

float calc_zeta(float x, float q) {
  constexpr float MACHEP = 1.11022302462515654042E-16;
  const float ZETA_EXPANSION[] = {
      12.0,
      -720.0,
      30240.0,
      -1209600.0,
      47900160.0,
      -1.8924375803183791606e9,
      7.47242496e10,
      -2.950130727918164224e12,
      1.1646782814350067249e14,
      -4.5979787224074726105e15,
      1.8152105401943546773e17,
      -7.1661652561756670113e18};
  if (x == 1.0f) {
    return INFINITY;
  }

  if (x < 1.0f) {
    return NAN;
  }

  if (q <= 0.0f) {
    if (q == trunc(q)) {
      return INFINITY;
    }
    if (x != trunc(x)) {
      return NAN;
    }
  }

  float s = pow(q, -x);
  float a = q;
  int i = 0;
  float b = 0.0f;
  while ((i < 9) || (a <= 9.0f)) {
    i += 1;
    a += 1.0f;
    b = pow(a, -x);
    s += b;
    if ((-MACHEP * s < b) && (b < MACHEP * s)) {
      return s;
    }
  }

  float w = a;
  s += b * w / (x - 1.0f);
  s -= 0.5f * b;
  a = 1.0f;
  float t;
  float k = 0.0f;
  for (int i = 0; i < 12; i++) {
    a *= x + k;
    b /= w;
    t = a * b / ZETA_EXPANSION[i];
    s += t;
    t = fabs(t / s);
    if (t < MACHEP) {
      return s;
    }
    k += 1.0f;
    a *= x + k;
    b /= w;
    k += 1.0f;
  }
  return s;
}

template <typename T0, typename T1>
kernel void lgamma(
    constant T0* input [[buffer(0)]],
    device T1* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
  output[id] = static_cast<T1>(LogGamma(static_cast<float>(input[id])));
}

template <typename T0, typename T1>
kernel void digamma(
    constant T0* input [[buffer(0)]],
    device T1* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
  float x = input[id];
  if (x < 0.0f) {
    if (x == trunc(x)) {
      // As per C++ standard for gamma related functions and SciPy,
      // If the argument is a negative integer, NaN is returned
      output[id] = static_cast<T1>(NAN);
    } else {
      // Extracts the fractional part of x as r, since tan(pi * r) is more
      // numerically accurate than tan(pi * x). While these operations are
      // mathematically equivalent since both x and r are in radians and tan()
      // has a periodicity of pi, in practice the computation of pi * x is a
      // source of error (when |x| > 1).
      float r = fract(x);
      output[id] = static_cast<T1>(
          calc_digamma_positive_domain(1.0f - x) - M_PI_F / tan(M_PI_F * r));
    }
  } else if (x == 0.0f) {
    // As per C++ standard for gamma related functions and SciPy,
    // If the argument is ±0, ±∞ is returned
    output[id] = static_cast<T1>(copysign(INFINITY, -x));
  } else {
    output[id] = static_cast<T1>(calc_digamma_positive_domain(x));
  }
}

template <typename T0, typename T1>
kernel void trigamma(
    constant T0* input [[buffer(0)]],
    device T1* output [[buffer(1)]],
    uint id [[thread_position_in_grid]]) {
  float x = input[id];
  output[id] = static_cast<T1>(calc_trigamma(x));
}

template <typename T0, typename T1>
kernel void polygamma(
    constant T0* input [[buffer(0)]],
    device T1* output [[buffer(1)]],
    constant int64_t& order [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
  // already blocked if n <= 1
  float x = input[id];
  float n = order;
  float sgn = ((order % 2) ? 1 : -1);
  output[id] = static_cast<T1>(sgn * Gamma(n + 1) * calc_zeta(n + 1, x));
}

#define INSTANTIATE_GAMMA_KERNELS(DTYPE0, DTYPE1)                             \
  template [[host_name("lgamma_" #DTYPE0 "_" #DTYPE1)]] kernel void lgamma(   \
      constant DTYPE0* input [[buffer(0)]],                                   \
      device DTYPE1* output [[buffer(1)]],                                    \
      uint id [[thread_position_in_grid]]);                                   \
  template [[host_name("digamma_" #DTYPE0 "_" #DTYPE1)]] kernel void digamma( \
      constant DTYPE0* input [[buffer(0)]],                                   \
      device DTYPE1* output [[buffer(1)]],                                    \
      uint id [[thread_position_in_grid]]);                                   \
  template [[host_name("trigamma_" #DTYPE0 "_" #DTYPE1)]] kernel void         \
  trigamma(                                                                   \
      constant DTYPE0* input [[buffer(0)]],                                   \
      device DTYPE1* output [[buffer(1)]],                                    \
      uint id [[thread_position_in_grid]]);                                   \
  template [[host_name("polygamma_" #DTYPE0 "_" #DTYPE1)]] kernel void        \
  polygamma(                                                                  \
      constant DTYPE0* input [[buffer(0)]],                                   \
      device DTYPE1* output [[buffer(1)]],                                    \
      constant int64_t& order [[buffer(2)]],                                  \
      uint id [[thread_position_in_grid]]);

#if __METAL_VERSION__ >= 310
INSTANTIATE_GAMMA_KERNELS(bfloat, bfloat);
#endif
INSTANTIATE_GAMMA_KERNELS(half, half);
INSTANTIATE_GAMMA_KERNELS(float, float);
INSTANTIATE_GAMMA_KERNELS(bool, float);
INSTANTIATE_GAMMA_KERNELS(uchar, float);
INSTANTIATE_GAMMA_KERNELS(char, float);
INSTANTIATE_GAMMA_KERNELS(short, float);
INSTANTIATE_GAMMA_KERNELS(int, float);
INSTANTIATE_GAMMA_KERNELS(long, float);
