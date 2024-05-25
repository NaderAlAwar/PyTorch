# Owner(s): ["module: dynamo"]

# this file is autogenerated via gen_ufuncs.py
# do not edit manually!
import numpy as np

from torch._numpy._ufuncs import *  # noqa: F403
from torch._numpy.testing import assert_allclose
from torch.testing._internal.common_utils import run_tests, TestCase


class TestBinaryUfuncBasic(TestCase):
    def test_add(self):
        assert_allclose(np.add(0.5, 0.6), add(0.5, 0.6), atol=1e-7, check_dtype=False)

    def test_arctan2(self):
        assert_allclose(
            np.arctan2(0.5, 0.6), arctan2(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_bitwise_and(self):
        assert_allclose(
            np.bitwise_and(5, 6), bitwise_and(5, 6), atol=1e-7, check_dtype=False
        )

    def test_bitwise_or(self):
        assert_allclose(
            np.bitwise_or(5, 6), bitwise_or(5, 6), atol=1e-7, check_dtype=False
        )

    def test_bitwise_xor(self):
        assert_allclose(
            np.bitwise_xor(5, 6), bitwise_xor(5, 6), atol=1e-7, check_dtype=False
        )

    def test_copysign(self):
        assert_allclose(
            np.copysign(0.5, 0.6), copysign(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_divide(self):
        assert_allclose(
            np.divide(0.5, 0.6), divide(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_equal(self):
        assert_allclose(
            np.equal(0.5, 0.6), equal(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_float_power(self):
        assert_allclose(
            np.float_power(0.5, 0.6),
            float_power(0.5, 0.6),
            atol=1e-7,
            check_dtype=False,
        )

    def test_floor_divide(self):
        assert_allclose(
            np.floor_divide(0.5, 0.6),
            floor_divide(0.5, 0.6),
            atol=1e-7,
            check_dtype=False,
        )

    def test_fmax(self):
        assert_allclose(np.fmax(0.5, 0.6), fmax(0.5, 0.6), atol=1e-7, check_dtype=False)

    def test_fmin(self):
        assert_allclose(np.fmin(0.5, 0.6), fmin(0.5, 0.6), atol=1e-7, check_dtype=False)

    def test_fmod(self):
        assert_allclose(np.fmod(0.5, 0.6), fmod(0.5, 0.6), atol=1e-7, check_dtype=False)

    def test_gcd(self):
        assert_allclose(np.gcd(5, 6), gcd(5, 6), atol=1e-7, check_dtype=False)

    def test_greater(self):
        assert_allclose(
            np.greater(0.5, 0.6), greater(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_greater_equal(self):
        assert_allclose(
            np.greater_equal(0.5, 0.6),
            greater_equal(0.5, 0.6),
            atol=1e-7,
            check_dtype=False,
        )

    def test_heaviside(self):
        assert_allclose(
            np.heaviside(0.5, 0.6), heaviside(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_hypot(self):
        assert_allclose(
            np.hypot(0.5, 0.6), hypot(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_lcm(self):
        assert_allclose(np.lcm(5, 6), lcm(5, 6), atol=1e-7, check_dtype=False)

    def test_ldexp(self):
        assert_allclose(np.ldexp(0.5, 6), ldexp(0.5, 6), atol=1e-7, check_dtype=False)

    def test_left_shift(self):
        assert_allclose(
            np.left_shift(5, 6), left_shift(5, 6), atol=1e-7, check_dtype=False
        )

    def test_less(self):
        assert_allclose(np.less(0.5, 0.6), less(0.5, 0.6), atol=1e-7, check_dtype=False)

    def test_less_equal(self):
        assert_allclose(
            np.less_equal(0.5, 0.6), less_equal(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_logaddexp(self):
        assert_allclose(
            np.logaddexp(0.5, 0.6), logaddexp(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_logaddexp2(self):
        assert_allclose(
            np.logaddexp2(0.5, 0.6), logaddexp2(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_logical_and(self):
        assert_allclose(
            np.logical_and(0.5, 0.6),
            logical_and(0.5, 0.6),
            atol=1e-7,
            check_dtype=False,
        )

    def test_logical_or(self):
        assert_allclose(
            np.logical_or(0.5, 0.6), logical_or(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_logical_xor(self):
        assert_allclose(
            np.logical_xor(0.5, 0.6),
            logical_xor(0.5, 0.6),
            atol=1e-7,
            check_dtype=False,
        )

    def test_matmul(self):
        assert_allclose(
            np.matmul([0.5], [0.6]), matmul([0.5], [0.6]), atol=1e-7, check_dtype=False
        )

    def test_maximum(self):
        assert_allclose(
            np.maximum(0.5, 0.6), maximum(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_minimum(self):
        assert_allclose(
            np.minimum(0.5, 0.6), minimum(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_remainder(self):
        assert_allclose(
            np.remainder(0.5, 0.6), remainder(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_multiply(self):
        assert_allclose(
            np.multiply(0.5, 0.6), multiply(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_nextafter(self):
        assert_allclose(
            np.nextafter(0.5, 0.6), nextafter(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_not_equal(self):
        assert_allclose(
            np.not_equal(0.5, 0.6), not_equal(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_power(self):
        assert_allclose(
            np.power(0.5, 0.6), power(0.5, 0.6), atol=1e-7, check_dtype=False
        )

    def test_right_shift(self):
        assert_allclose(
            np.right_shift(5, 6), right_shift(5, 6), atol=1e-7, check_dtype=False
        )

    def test_subtract(self):
        assert_allclose(
            np.subtract(0.5, 0.6), subtract(0.5, 0.6), atol=1e-7, check_dtype=False
        )


if __name__ == "__main__":
    run_tests()
