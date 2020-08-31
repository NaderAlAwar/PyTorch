import torch
import unittest
import math
from contextlib import contextmanager
from itertools import product

from torch.testing._internal.common_utils import \
    (TestCase, run_tests, TEST_NUMPY, TEST_LIBROSA)
from torch.testing._internal.common_device_type import \
    (instantiate_device_type_tests, dtypes, onlyOnCPUAndCUDA, precisionOverride,
     skipCPUIfNoMkl, skipCUDAIfRocm, deviceCountAtLeast, onlyCUDA)

if TEST_NUMPY:
    import numpy as np


if TEST_LIBROSA:
    import librosa

# saves the torch.fft function that's clobbered by importing the torch.fft module
fft_fn = torch.fft
import torch.fft


def _complex_stft(x, *args, **kwargs):
    # Transform real and imaginary components separably
    stft_real = torch.stft(x.real, *args, **kwargs, return_complex=True, onesided=False)
    stft_imag = torch.stft(x.imag, *args, **kwargs, return_complex=True, onesided=False)
    return stft_real + 1j * stft_imag


def _hermitian_conj(x, dim):
    out = torch.empty_like(x)
    mid = (x.size(dim) - 1) // 2
    idx = [slice(None)] * out.dim()
    idx_center = list(idx)
    idx_center[dim] = 0
    out[idx] = x[idx]

    idx_neg = list(idx)
    idx_neg[dim] = slice(-mid, None)
    idx_pos = idx
    idx_pos[dim] = slice(1, mid+1)

    out[idx_pos] = x[idx_neg].flip(dim)
    out[idx_neg] = x[idx_pos].flip(dim)
    if (2*mid + 1 < x.size(dim)):
        idx[dim] = mid + 1
        out[idx] = x[idx]
    return out.conj()


def _complex_istft(x, *args, **kwargs):
    # Decompose into Hermitian (FFT of real) and anti-Hermitian (FFT of imaginary)
    n_fft = x.size(-2)
    slc = (Ellipsis, slice(None, n_fft//2 + 1), slice(None))

    hconj = _hermitian_conj(x, dim=-2)
    x_hermitian = (x + hconj) / 2
    x_antihermitian = (x - hconj) / 2
    istft_real = torch.istft(x_hermitian[slc], *args, **kwargs, onesided=True)
    istft_imag = torch.istft(-1j * x_antihermitian[slc], *args, **kwargs, onesided=True)
    return istft_real + 1j * istft_imag


def stft_definition(x, hop_length, window):
    n_fft = window.numel()
    X = torch.empty((n_fft, (x.numel() - n_fft + hop_length) // hop_length),
                    device=x.device, dtype=torch.cdouble)
    for m in range(X.size(1)):
        if m*hop_length + n_fft > x.numel():
            slc = torch.empty(n_fft, device=x.device, dtype=x.dtype)
            tmp = x[m*hop_length:]
            slc[:tmp.numel()] = tmp
        else:
            slc = x[m*hop_length: m*hop_length + n_fft]
        X[:, m] = torch.fft.fft(slc * window)
    return X


# Tests of functions related to Fourier analysis in the torch.fft namespace
class TestFFT(TestCase):
    exact_dtype = True

    @skipCPUIfNoMkl
    @skipCUDAIfRocm
    def test_fft_function_clobbered(self, device):
        t = torch.randn((100, 2), device=device)
        eager_result = fft_fn(t, 1)

        def method_fn(t):
            return t.fft(1)
        scripted_method_fn = torch.jit.script(method_fn)

        self.assertEqual(scripted_method_fn(t), eager_result)

        with self.assertRaisesRegex(TypeError, "'module' object is not callable"):
            torch.fft(t, 1)

    @skipCPUIfNoMkl
    @skipCUDAIfRocm
    @unittest.skipIf(not TEST_NUMPY, 'NumPy not found')
    @precisionOverride({torch.complex64: 1e-4})
    @dtypes(torch.complex64, torch.complex128)
    def test_fft(self, device, dtype):
        test_inputs = (torch.randn(67, device=device, dtype=dtype),
                       torch.randn(4029, device=device, dtype=dtype))

        def fn(t):
            return torch.fft.fft(t)
        scripted_fn = torch.jit.script(fn)

        # TODO: revisit the following function if t.fft() becomes torch.fft.fft
        # def method_fn(t):
        #     return t.fft()
        # scripted_method_fn = torch.jit.script(method_fn)

        # TODO: revisit the following function if t.fft() becomes torch.fft.fft
        # torch_fns = (torch.fft.fft, torch.Tensor.fft, scripted_fn, scripted_method_fn)
        torch_fns = (torch.fft.fft, scripted_fn)

        for input in test_inputs:
            expected = np.fft.fft(input.cpu().numpy())
            for fn in torch_fns:
                actual = fn(input)
                self.assertEqual(actual, expected, exact_dtype=(dtype is torch.complex128))

    # Note: NumPy will throw a ValueError for an empty input
    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @dtypes(torch.complex64, torch.complex128)
    def test_empty_fft(self, device, dtype):
        t = torch.empty(0, device=device, dtype=dtype)

        if self.device_type == 'cuda':
            with self.assertRaisesRegex(RuntimeError, "cuFFT error"):
                torch.fft.fft(t)
            return

        # CPU (MKL)
        with self.assertRaisesRegex(RuntimeError, "MKL FFT error"):
            torch.fft.fft(t)

    @dtypes(torch.int64, torch.float32)
    def test_fft_invalid_dtypes(self, device, dtype):
        if dtype.is_floating_point:
            t = torch.randn(64, device=device, dtype=dtype)
        else:
            t = torch.randint(-2, 2, (64,), device=device, dtype=dtype)

        with self.assertRaisesRegex(RuntimeError, "Expected a complex tensor"):
            torch.fft.fft(t)

    # Legacy fft tests
    def _test_fft_ifft_rfft_irfft(self, device, dtype):
        def _test_complex(sizes, signal_ndim, prepro_fn=lambda x: x):
            x = prepro_fn(torch.randn(*sizes, dtype=dtype, device=device))
            for normalized in (True, False):
                res = x.fft(signal_ndim, normalized=normalized)
                rec = res.ifft(signal_ndim, normalized=normalized)
                self.assertEqual(x, rec, atol=1e-8, rtol=0, msg='fft and ifft')
                res = x.ifft(signal_ndim, normalized=normalized)
                rec = res.fft(signal_ndim, normalized=normalized)
                self.assertEqual(x, rec, atol=1e-8, rtol=0, msg='ifft and fft')

        def _test_real(sizes, signal_ndim, prepro_fn=lambda x: x):
            x = prepro_fn(torch.randn(*sizes, dtype=dtype, device=device))
            signal_numel = 1
            signal_sizes = x.size()[-signal_ndim:]
            for normalized, onesided in product((True, False), repeat=2):
                res = x.rfft(signal_ndim, normalized=normalized, onesided=onesided)
                if not onesided:  # check Hermitian symmetry
                    def test_one_sample(res, test_num=10):
                        idxs_per_dim = [torch.LongTensor(test_num).random_(s).tolist() for s in signal_sizes]
                        for idx in zip(*idxs_per_dim):
                            reflected_idx = tuple((s - i) % s for i, s in zip(idx, res.size()))
                            idx_val = res.__getitem__(idx)
                            reflected_val = res.__getitem__(reflected_idx)
                            self.assertEqual(idx_val[0], reflected_val[0], msg='rfft hermitian symmetry on real part')
                            self.assertEqual(idx_val[1], -reflected_val[1], msg='rfft hermitian symmetry on imaginary part')
                    if len(sizes) == signal_ndim:
                        test_one_sample(res)
                    else:
                        output_non_batch_shape = res.size()[-(signal_ndim + 1):]
                        flatten_batch_res = res.view(-1, *output_non_batch_shape)
                        nb = flatten_batch_res.size(0)
                        test_idxs = torch.LongTensor(min(nb, 4)).random_(nb)
                        for test_idx in test_idxs.tolist():
                            test_one_sample(flatten_batch_res[test_idx])
                    # compare with C2C
                    xc = torch.stack([x, torch.zeros_like(x)], -1)
                    xc_res = xc.fft(signal_ndim, normalized=normalized)
                    self.assertEqual(res, xc_res)
                test_input_signal_sizes = [signal_sizes]
                rec = res.irfft(signal_ndim, normalized=normalized,
                                onesided=onesided, signal_sizes=signal_sizes)
                self.assertEqual(x, rec, atol=1e-8, rtol=0, msg='rfft and irfft')
                if not onesided:  # check that we can use C2C ifft
                    rec = res.ifft(signal_ndim, normalized=normalized)
                    self.assertEqual(x, rec.select(-1, 0), atol=1e-8, rtol=0, msg='twosided rfft and ifft real')
                    self.assertEqual(rec.select(-1, 1).abs().mean(), 0, atol=1e-8,
                                     rtol=0, msg='twosided rfft and ifft imaginary')

        # contiguous case
        _test_real((100,), 1)
        _test_real((10, 1, 10, 100), 1)
        _test_real((100, 100), 2)
        _test_real((2, 2, 5, 80, 60), 2)
        _test_real((50, 40, 70), 3)
        _test_real((30, 1, 50, 25, 20), 3)

        _test_complex((100, 2), 1)
        _test_complex((100, 100, 2), 1)
        _test_complex((100, 100, 2), 2)
        _test_complex((1, 20, 80, 60, 2), 2)
        _test_complex((50, 40, 70, 2), 3)
        _test_complex((6, 5, 50, 25, 20, 2), 3)

        # non-contiguous case
        _test_real((165,), 1, lambda x: x.narrow(0, 25, 100))  # input is not aligned to complex type
        _test_real((100, 100, 3), 1, lambda x: x[:, :, 0])
        _test_real((100, 100), 2, lambda x: x.t())
        _test_real((20, 100, 10, 10), 2, lambda x: x.view(20, 100, 100)[:, :60])
        _test_real((65, 80, 115), 3, lambda x: x[10:60, 13:53, 10:80])
        _test_real((30, 20, 50, 25), 3, lambda x: x.transpose(1, 2).transpose(2, 3))

        _test_complex((2, 100), 1, lambda x: x.t())
        _test_complex((100, 2), 1, lambda x: x.expand(100, 100, 2))
        _test_complex((300, 200, 3), 2, lambda x: x[:100, :100, 1:])  # input is not aligned to complex type
        _test_complex((20, 90, 110, 2), 2, lambda x: x[:, 5:85].narrow(2, 5, 100))
        _test_complex((40, 60, 3, 80, 2), 3, lambda x: x.transpose(2, 0).select(0, 2)[5:55, :, 10:])
        _test_complex((30, 55, 50, 22, 2), 3, lambda x: x[:, 3:53, 15:40, 1:21])

        # non-contiguous with strides not representable as aligned with complex type
        _test_complex((50,), 1, lambda x: x.as_strided([5, 5, 2], [3, 2, 1]))
        _test_complex((50,), 1, lambda x: x.as_strided([5, 5, 2], [4, 2, 2]))
        _test_complex((50,), 1, lambda x: x.as_strided([5, 5, 2], [4, 3, 1]))
        _test_complex((50,), 2, lambda x: x.as_strided([5, 5, 2], [3, 3, 1]))
        _test_complex((50,), 2, lambda x: x.as_strided([5, 5, 2], [4, 2, 2]))
        _test_complex((50,), 2, lambda x: x.as_strided([5, 5, 2], [4, 3, 1]))

    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    @onlyOnCPUAndCUDA
    @dtypes(torch.double)
    def test_fft_ifft_rfft_irfft(self, device, dtype):
        self._test_fft_ifft_rfft_irfft(device, dtype)

    @deviceCountAtLeast(1)
    @skipCUDAIfRocm
    @onlyCUDA
    @dtypes(torch.double)
    def test_cufft_plan_cache(self, devices, dtype):
        @contextmanager
        def plan_cache_max_size(device, n):
            if device is None:
                plan_cache = torch.backends.cuda.cufft_plan_cache
            else:
                plan_cache = torch.backends.cuda.cufft_plan_cache[device]
            original = plan_cache.max_size
            plan_cache.max_size = n
            yield
            plan_cache.max_size = original

        with plan_cache_max_size(devices[0], max(1, torch.backends.cuda.cufft_plan_cache.size - 10)):
            self._test_fft_ifft_rfft_irfft(devices[0], dtype)

        with plan_cache_max_size(devices[0], 0):
            self._test_fft_ifft_rfft_irfft(devices[0], dtype)

        torch.backends.cuda.cufft_plan_cache.clear()

        # check that stll works after clearing cache
        with plan_cache_max_size(devices[0], 10):
            self._test_fft_ifft_rfft_irfft(devices[0], dtype)

        with self.assertRaisesRegex(RuntimeError, r"must be non-negative"):
            torch.backends.cuda.cufft_plan_cache.max_size = -1

        with self.assertRaisesRegex(RuntimeError, r"read-only property"):
            torch.backends.cuda.cufft_plan_cache.size = -1

        with self.assertRaisesRegex(RuntimeError, r"but got device with index"):
            torch.backends.cuda.cufft_plan_cache[torch.cuda.device_count() + 10]

        # Multigpu tests
        if len(devices) > 1:
            # Test that different GPU has different cache
            x0 = torch.randn(2, 3, 3, device=devices[0])
            x1 = x0.to(devices[1])
            self.assertEqual(x0.rfft(2), x1.rfft(2))
            # If a plan is used across different devices, the following line (or
            # the assert above) would trigger illegal memory access. Other ways
            # to trigger the error include
            #   (1) setting CUDA_LAUNCH_BLOCKING=1 (pytorch/pytorch#19224) and
            #   (2) printing a device 1 tensor.
            x0.copy_(x1)

            # Test that un-indexed `torch.backends.cuda.cufft_plan_cache` uses current device
            with plan_cache_max_size(devices[0], 10):
                with plan_cache_max_size(devices[1], 11):
                    self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                    self.assertEqual(torch.backends.cuda.cufft_plan_cache[1].max_size, 11)

                    self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0
                    with torch.cuda.device(devices[1]):
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1
                        with torch.cuda.device(devices[0]):
                            self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0

                self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                with torch.cuda.device(devices[1]):
                    with plan_cache_max_size(None, 11):  # default is cuda:1
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache[0].max_size, 10)
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache[1].max_size, 11)

                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1
                        with torch.cuda.device(devices[0]):
                            self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 10)  # default is cuda:0
                        self.assertEqual(torch.backends.cuda.cufft_plan_cache.max_size, 11)  # default is cuda:1

    # passes on ROCm w/ python 2.7, fails w/ python 3.6
    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    @dtypes(torch.double)
    def test_stft(self, device, dtype):
        if not TEST_LIBROSA:
            raise unittest.SkipTest('librosa not found')

        def librosa_stft(x, n_fft, hop_length, win_length, window, center):
            if window is None:
                window = np.ones(n_fft if win_length is None else win_length)
            else:
                window = window.cpu().numpy()
            input_1d = x.dim() == 1
            if input_1d:
                x = x.view(1, -1)
            result = []
            for xi in x:
                ri = librosa.stft(xi.cpu().numpy(), n_fft, hop_length, win_length, window, center=center)
                result.append(torch.from_numpy(np.stack([ri.real, ri.imag], -1)))
            result = torch.stack(result, 0)
            if input_1d:
                result = result[0]
            return result

        def _test(sizes, n_fft, hop_length=None, win_length=None, win_sizes=None,
                  center=True, expected_error=None):
            x = torch.randn(*sizes, dtype=dtype, device=device)
            if win_sizes is not None:
                window = torch.randn(*win_sizes, dtype=dtype, device=device)
            else:
                window = None
            if expected_error is None:
                result = x.stft(n_fft, hop_length, win_length, window, center=center)
                # NB: librosa defaults to np.complex64 output, no matter what
                # the input dtype
                ref_result = librosa_stft(x, n_fft, hop_length, win_length, window, center)
                self.assertEqual(result, ref_result, atol=7e-6, rtol=0, msg='stft comparison against librosa', exact_dtype=False)
                # With return_complex=True, the result is the same but viewed as complex instead of real
                result_complex = x.stft(n_fft, hop_length, win_length, window, center=center, return_complex=True)
                self.assertEqual(result_complex, torch.view_as_complex(result))
            else:
                self.assertRaises(expected_error,
                                  lambda: x.stft(n_fft, hop_length, win_length, window, center=center))

        for center in [True, False]:
            _test((10,), 7, center=center)
            _test((10, 4000), 1024, center=center)

            _test((10,), 7, 2, center=center)
            _test((10, 4000), 1024, 512, center=center)

            _test((10,), 7, 2, win_sizes=(7,), center=center)
            _test((10, 4000), 1024, 512, win_sizes=(1024,), center=center)

            # spectral oversample
            _test((10,), 7, 2, win_length=5, center=center)
            _test((10, 4000), 1024, 512, win_length=100, center=center)

        _test((10, 4, 2), 1, 1, expected_error=RuntimeError)
        _test((10,), 11, 1, center=False, expected_error=RuntimeError)
        _test((10,), -1, 1, expected_error=RuntimeError)
        _test((10,), 3, win_length=5, expected_error=RuntimeError)
        _test((10,), 5, 4, win_sizes=(11,), expected_error=RuntimeError)
        _test((10,), 5, 4, win_sizes=(1, 1), expected_error=RuntimeError)


    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    @dtypes(torch.double, torch.cdouble)
    def test_complex_stft_roundtrip(self, device, dtype):
        test_args = list(product(
            # input
            (torch.randn(600, device=device, dtype=dtype),
             torch.randn(807, device=device, dtype=dtype),
             torch.randn(12, 14, device=device, dtype=dtype),
             torch.randn(9, 6, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (None, 10),
            # center
            (True,),
            # pad_mode
            ("constant",),
            # normalized
            (True, False),
            # onesided
            (True, False) if not dtype.is_complex else (False,),
        ))

        for args in test_args:
            x = args[0]
            x_stft = torch.stft(
                x, n_fft=args[1], hop_length=args[2],
                center=args[3], pad_mode=args[4], normalized=args[5],
                onesided=args[6], return_complex=True)
            x_roundtrip = torch.istft(
                x_stft, n_fft=args[1], hop_length=args[2], center=args[3],
                normalized=args[5], onesided=args[6], length=x.size(-1),
                return_complex=dtype.is_complex)
            self.assertEqual(x_roundtrip, x)

    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    @dtypes(torch.double, torch.cdouble)
    def test_stft_roundtrip_complex_window(self, device, dtype):
        test_args = list(product(
            # input
            (torch.randn(600, device=device, dtype=dtype),
             torch.randn(807, device=device, dtype=dtype),
             torch.randn(12, 14, device=device, dtype=dtype),
             torch.randn(9, 6, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (None, 10),
            # pad_mode
            ("constant",),
            # normalized
            (True, False),
        ))
        for args in test_args:
            x, n_fft, hop_length, pad_mode, normalized = args
            window = torch.rand(n_fft, device=device, dtype=torch.cdouble)
            x_stft = torch.stft(
                x, n_fft=n_fft, hop_length=hop_length, window=window,
                center=True, pad_mode=pad_mode, normalized=normalized)
            self.assertEqual(x_stft.dtype, torch.cdouble)
            self.assertEqual(x_stft.size(-2), n_fft)  # Not onesided

            x_roundtrip = torch.istft(
                x_stft, n_fft=n_fft, hop_length=hop_length, window=window,
                center=True, normalized=normalized, length=x.size(-1),
                return_complex=True)
            self.assertEqual(x_stft.dtype, torch.cdouble)

            if not dtype.is_complex:
                self.assertEqual(x_roundtrip.imag, torch.zeros_like(x_roundtrip.imag),
                                 atol=1e-6, rtol=0)
                self.assertEqual(x_roundtrip.real, x)
            else:
                self.assertEqual(x_roundtrip, x)


    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    @dtypes(torch.cdouble)
    def test_complex_stft_definition(self, device, dtype):
        test_args = list(product(
            # input
            (torch.randn(600, device=device, dtype=dtype),
             torch.randn(807, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (10, 15)
        ))

        for args in test_args:
            window = torch.randn(args[1], device=device, dtype=dtype)
            expected = stft_definition(args[0], args[2], window)
            actual = torch.stft(*args, window=window, center=False)
            self.assertEqual(actual, expected)

    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    @dtypes(torch.cdouble)
    def test_complex_stft_real_equiv(self, device, dtype):
        test_args = list(product(
            # input
            (torch.rand(600, device=device, dtype=dtype),
             torch.rand(807, device=device, dtype=dtype),
             torch.rand(14, 50, device=device, dtype=dtype),
             torch.rand(6, 51, device=device, dtype=dtype)),
            # n_fft
            (50, 27),
            # hop_length
            (None, 10),
            # win_length
            (None, 20),
            # center
            (False, True),
            # pad_mode
            ("constant",),
            # normalized
            (True, False),
        ))

        for args in test_args:
            x, n_fft, hop_length, win_length, center, pad_mode, normalized = args
            expected = _complex_stft(x, n_fft, hop_length=hop_length,
                                     win_length=win_length, pad_mode=pad_mode,
                                     center=center, normalized=normalized)
            actual = torch.stft(x, n_fft, hop_length=hop_length,
                                win_length=win_length, pad_mode=pad_mode,
                                center=center, normalized=normalized)
            self.assertEqual(expected, actual)

    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    @dtypes(torch.cdouble)
    def test_complex_istft_real_equiv(self, device, dtype):
        test_args = list(product(
            # input
            (torch.rand(40, 20, device=device, dtype=dtype),
             torch.rand(25, 1, device=device, dtype=dtype),
             torch.rand(4, 20, 10, device=device, dtype=dtype)),
            # hop_length
            (None, 10),
            # center
            (False, True),
            # normalized
            (True, False),
        ))

        for args in test_args:
            x, hop_length, center, normalized = args
            n_fft = x.size(-2)
            expected = _complex_istft(x, n_fft, hop_length=hop_length,
                                      center=center, normalized=normalized)
            actual = torch.istft(x, n_fft, hop_length=hop_length,
                                 center=center, normalized=normalized,
                                 return_complex=True)
            self.assertEqual(expected, actual)

    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    def test_complex_stft_onesided(self, device):
        # stft of complex input cannot be onesided
        for x_dtype, window_dtype in product((torch.double, torch.cdouble), repeat=2):
            x = torch.rand(100, device=device, dtype=x_dtype)
            window = torch.rand(10, device=device, dtype=window_dtype)

            if x_dtype.is_complex or window_dtype.is_complex:
                with self.assertRaisesRegex(RuntimeError, 'complex'):
                    x.stft(10, window=window, pad_mode='constant', onesided=True)
            else:
                y = x.stft(10, window=window, pad_mode='constant', onesided=True)
                self.assertEqual(y.dtype, torch.double)
                self.assertEqual(y.size(), (6, 51, 2))

        y = torch.rand(100, device=device, dtype=torch.double)
        window = torch.randn(10, device=device, dtype=torch.cdouble)
        with self.assertRaisesRegex(RuntimeError, 'complex'):
            x.stft(10, pad_mode='constant', onesided=True)

    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    def test_fft_input_modification(self, device):
        # FFT functions should not modify their input (gh-34551)

        signal = torch.ones((2, 2, 2), device=device)
        signal_copy = signal.clone()
        spectrum = signal.fft(2)
        self.assertEqual(signal, signal_copy)

        spectrum_copy = spectrum.clone()
        _ = torch.ifft(spectrum, 2)
        self.assertEqual(spectrum, spectrum_copy)

        half_spectrum = torch.rfft(signal, 2)
        self.assertEqual(signal, signal_copy)

        half_spectrum_copy = half_spectrum.clone()
        _ = torch.irfft(half_spectrum_copy, 2, signal_sizes=(2, 2))
        self.assertEqual(half_spectrum, half_spectrum_copy)

    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    @dtypes(torch.double)
    def test_istft_round_trip_simple_cases(self, device, dtype):
        """stft -> istft should recover the original signale"""
        def _test(input, n_fft, length):
            stft = torch.stft(input, n_fft=n_fft)
            inverse = torch.istft(stft, n_fft=n_fft, length=length)
            self.assertEqual(input, inverse, exact_dtype=True)

        _test(torch.ones(4, dtype=dtype, device=device), 4, 4)
        _test(torch.zeros(4, dtype=dtype, device=device), 4, 4)

    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    @dtypes(torch.double)
    def test_istft_round_trip_various_params(self, device, dtype):
        """stft -> istft should recover the original signale"""
        def _test_istft_is_inverse_of_stft(stft_kwargs):
            # generates a random sound signal for each tril and then does the stft/istft
            # operation to check whether we can reconstruct signal
            data_sizes = [(2, 20), (3, 15), (4, 10)]
            num_trials = 100
            istft_kwargs = stft_kwargs.copy()
            del istft_kwargs['pad_mode']
            for sizes in data_sizes:
                for i in range(num_trials):
                    original = torch.randn(*sizes, dtype=dtype, device=device)
                    stft = torch.stft(original, **stft_kwargs)
                    inversed = torch.istft(stft, length=original.size(1), **istft_kwargs)

                    # trim the original for case when constructed signal is shorter than original
                    original = original[..., :inversed.size(-1)]
                    self.assertEqual(
                        inversed, original, msg='istft comparison against original',
                        atol=7e-6, rtol=0, exact_dtype=True)

        patterns = [
            # hann_window, centered, normalized, onesided
            {
                'n_fft': 12,
                'hop_length': 4,
                'win_length': 12,
                'window': torch.hann_window(12, dtype=dtype, device=device),
                'center': True,
                'pad_mode': 'reflect',
                'normalized': True,
                'onesided': True,
            },
            # hann_window, centered, not normalized, not onesided
            {
                'n_fft': 12,
                'hop_length': 2,
                'win_length': 8,
                'window': torch.hann_window(8, dtype=dtype, device=device),
                'center': True,
                'pad_mode': 'reflect',
                'normalized': False,
                'onesided': False,
            },
            # hamming_window, centered, normalized, not onesided
            {
                'n_fft': 15,
                'hop_length': 3,
                'win_length': 11,
                'window': torch.hamming_window(11, dtype=dtype, device=device),
                'center': True,
                'pad_mode': 'constant',
                'normalized': True,
                'onesided': False,
            },
            # hamming_window, not centered, not normalized, onesided
            # window same size as n_fft
            {
                'n_fft': 5,
                'hop_length': 2,
                'win_length': 5,
                'window': torch.hamming_window(5, dtype=dtype, device=device),
                'center': False,
                'pad_mode': 'constant',
                'normalized': False,
                'onesided': True,
            },
            # hamming_window, not centered, not normalized, not onesided
            # window same size as n_fft
            {
                'n_fft': 3,
                'hop_length': 2,
                'win_length': 3,
                'window': torch.hamming_window(3, dtype=dtype, device=device),
                'center': False,
                'pad_mode': 'reflect',
                'normalized': False,
                'onesided': False,
            },
        ]
        for i, pattern in enumerate(patterns):
            _test_istft_is_inverse_of_stft(pattern)

    @onlyOnCPUAndCUDA
    def test_istft_throws(self, device):
        """istft should throw exception for invalid parameters"""
        stft = torch.zeros((3, 5, 2), device=device)
        # the window is size 1 but it hops 20 so there is a gap which throw an error
        self.assertRaises(
            RuntimeError, torch.istft, stft, n_fft=4,
            hop_length=20, win_length=1, window=torch.ones(1))
        # A window of zeros does not meet NOLA
        invalid_window = torch.zeros(4, device=device)
        self.assertRaises(
            RuntimeError, torch.istft, stft, n_fft=4, win_length=4, window=invalid_window)
        # Input cannot be empty
        self.assertRaises(RuntimeError, torch.istft, torch.zeros((3, 0, 2)), 2)
        self.assertRaises(RuntimeError, torch.istft, torch.zeros((0, 3, 2)), 2)

    @onlyOnCPUAndCUDA
    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    @dtypes(torch.double)
    def test_istft_of_sine(self, device, dtype):
        def _test(amplitude, L, n):
            # stft of amplitude*sin(2*pi/L*n*x) with the hop length and window size equaling L
            x = torch.arange(2 * L + 1, device=device, dtype=dtype)
            original = amplitude * torch.sin(2 * math.pi / L * x * n)
            # stft = torch.stft(original, L, hop_length=L, win_length=L,
            #                   window=torch.ones(L), center=False, normalized=False)
            stft = torch.zeros((L // 2 + 1, 2, 2), device=device, dtype=dtype)
            stft_largest_val = (amplitude * L) / 2.0
            if n < stft.size(0):
                stft[n, :, 1] = -stft_largest_val

            if 0 <= L - n < stft.size(0):
                # symmetric about L // 2
                stft[L - n, :, 1] = stft_largest_val

            inverse = torch.istft(
                stft, L, hop_length=L, win_length=L,
                window=torch.ones(L, device=device, dtype=dtype), center=False, normalized=False)
            # There is a larger error due to the scaling of amplitude
            original = original[..., :inverse.size(-1)]
            self.assertEqual(inverse, original, atol=1e-3, rtol=0)

        _test(amplitude=123, L=5, n=1)
        _test(amplitude=150, L=5, n=2)
        _test(amplitude=111, L=5, n=3)
        _test(amplitude=160, L=7, n=4)
        _test(amplitude=145, L=8, n=5)
        _test(amplitude=80, L=9, n=6)
        _test(amplitude=99, L=10, n=7)

    @onlyOnCPUAndCUDA
    @skipCUDAIfRocm
    @skipCPUIfNoMkl
    @dtypes(torch.double)
    def test_istft_linearity(self, device, dtype):
        num_trials = 100

        def _test(data_size, kwargs):
            for i in range(num_trials):
                tensor1 = torch.randn(data_size, device=device, dtype=dtype)
                tensor2 = torch.randn(data_size, device=device, dtype=dtype)
                a, b = torch.rand(2, dtype=dtype, device=device)
                # Also compare method vs. functional call signature
                istft1 = tensor1.istft(**kwargs)
                istft2 = tensor2.istft(**kwargs)
                istft = a * istft1 + b * istft2
                estimate = torch.istft(a * tensor1 + b * tensor2, **kwargs)
                self.assertEqual(istft, estimate, atol=1e-5, rtol=0)
        patterns = [
            # hann_window, centered, normalized, onesided
            (
                (2, 7, 7, 2),
                {
                    'n_fft': 12,
                    'window': torch.hann_window(12, device=device, dtype=dtype),
                    'center': True,
                    'normalized': True,
                    'onesided': True,
                },
            ),
            # hann_window, centered, not normalized, not onesided
            (
                (2, 12, 7, 2),
                {
                    'n_fft': 12,
                    'window': torch.hann_window(12, device=device, dtype=dtype),
                    'center': True,
                    'normalized': False,
                    'onesided': False,
                },
            ),
            # hamming_window, centered, normalized, not onesided
            (
                (2, 12, 7, 2),
                {
                    'n_fft': 12,
                    'window': torch.hamming_window(12, device=device, dtype=dtype),
                    'center': True,
                    'normalized': True,
                    'onesided': False,
                },
            ),
            # hamming_window, not centered, not normalized, onesided
            (
                (2, 7, 3, 2),
                {
                    'n_fft': 12,
                    'window': torch.hamming_window(12, device=device, dtype=dtype),
                    'center': False,
                    'normalized': False,
                    'onesided': True,
                },
            )
        ]
        for data_size, kwargs in patterns:
            _test(data_size, kwargs)

    @onlyOnCPUAndCUDA
    @skipCPUIfNoMkl
    @skipCUDAIfRocm
    def test_batch_istft(self, device):
        original = torch.tensor([
            [[4., 0.], [4., 0.], [4., 0.], [4., 0.], [4., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]],
            [[0., 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.]]
        ], device=device)

        single = original.repeat(1, 1, 1, 1)
        multi = original.repeat(4, 1, 1, 1)

        i_original = torch.istft(original, n_fft=4, length=4)
        i_single = torch.istft(single, n_fft=4, length=4)
        i_multi = torch.istft(multi, n_fft=4, length=4)

        self.assertEqual(i_original.repeat(1, 1), i_single, atol=1e-6, rtol=0, exact_dtype=True)
        self.assertEqual(i_original.repeat(4, 1), i_multi, atol=1e-6, rtol=0, exact_dtype=True)

instantiate_device_type_tests(TestFFT, globals())

if __name__ == '__main__':
    run_tests()
