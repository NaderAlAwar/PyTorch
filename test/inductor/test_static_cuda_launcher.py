# Owner(s): ["module: inductor"]
import os
import tempfile

import torch
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor.runtime import triton_helpers
from torch._inductor.runtime.static_cuda_launcher import StaticallyLaunchedCudaKernel
from torch._inductor.runtime.triton_compat import CompiledKernel, tl, triton
from torch._inductor.runtime.triton_helpers import libdevice
from torch._inductor.test_case import TestCase
from torch.testing._internal.common_utils import skipIfRocm
from torch.testing._internal.triton_utils import requires_cuda


@requires_cuda
class TestStaticCudaLauncher(TestCase):
    def setUp(self):
        super().setUp()
        self.tmp_files = []

    def tearDown(self):
        super().tearDown()
        for tmp_file in self.tmp_files:
            try:
                os.remove(tmp_file.name)
            except OSError:
                pass

    def write_cubin_to_tmp(self, kernel: CompiledKernel) -> str:
        """
        Only used for tests where we don't have a cubin path.
        """
        if hasattr(kernel, "_cubin_path"):
            return
        # Just used by tests for now.
        # TODO: derive cubin_path from wherever triton stores the cubin file on disk.
        tmp_file = tempfile.NamedTemporaryFile(mode="wb", delete=False)
        with tmp_file:
            tmp_file.write(kernel.asm["cubin"])
        self.tmp_files.append(tmp_file)
        return tmp_file.name

    def _make_launcher(
        self,
        compiled_kernel: CompiledKernel,
    ) -> StaticallyLaunchedCudaKernel:
        """
        Compiles a Triton kernel with the provided *args,
        writes its cubin to the temporary file, and returns the file path.
        """
        cubin_file = self.write_cubin_to_tmp(compiled_kernel)
        compiled_kernel._cubin_path = cubin_file
        result = StaticallyLaunchedCudaKernel(compiled_kernel)
        result.load_kernel()
        return result

    @skipIfRocm
    def test_basic(self):
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        arg1 = 5
        args = (arg0, arg1)
        compiled_kernel = simple_kernel[(1,)](*args)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device="cuda"))
        self.assertEqual(launcher.arg_tys, "Oi")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run(1, 1, 1, stream, new_arg0, arg1)
        self.assertEqual(new_arg0, arg0)

    # I wish I could macro all int types this into a single unit test on a loop, but
    # 1. variables aren't allowed as type annotations in python
    # 2. triton relies on inspect.get_source to get the type annotations
    # so I can't even use exec() to generate the test cases.
    # So we'll just make a few kernels by hand
    @skipIfRocm
    def test_unsigned_integers(self):
        @triton.jit
        def unsigned_integers(
            arg0, arg1: tl.uint8, arg2: tl.uint16, arg3: tl.uint32, arg4: tl.uint64
        ):
            x = tl.load(arg0)
            y = arg1 + arg2 + arg3 + arg4
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.uint64, device="cuda")
        # Using small numbers creates a Literal type which triton treats as a constant
        args = (arg0, 50, 50, 50, 50)

        compiled_kernel = unsigned_integers[1,](*args)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([200], dtype=torch.uint64, device="cuda"))
        self.assertEqual(launcher.arg_tys, "OBHIK")
        new_arg0 = torch.zeros(1, dtype=torch.uint64, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(1, 1, 1, stream, new_arg0, 50, 50, 50, 50)
        self.assertEqual(new_arg0, arg0)

    @skipIfRocm
    def test_signed_integers(self):
        @triton.jit
        def signed_integers(
            arg0, arg1: tl.int8, arg2: tl.int16, arg3: tl.int32, arg4: tl.int64
        ):
            x = tl.load(arg0)
            y = arg1 + arg2 + arg3 + arg4
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int64, device="cuda")
        # Using small numbers creates a Literal type which triton treats as a constant
        args = (arg0, 50, 50, 50, 50)

        compiled_kernel = signed_integers[1,](*args)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([200], dtype=torch.int64, device="cuda"))
        self.assertEqual(launcher.arg_tys, "Obhil")
        new_arg0 = torch.zeros(1, dtype=torch.int64, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(1, 1, 1, stream, new_arg0, 50, 50, 50, 50)
        self.assertEqual(new_arg0, arg0)

    # TODO: floats don't work properly, triton seems to think they're all tl.float32
    # despite type annotations.
    # There's also not really a good way for me to make a float16 in python...
    @skipIfRocm
    def test_floats(self):
        @triton.jit
        def floats(arg0, arg1: tl.float16, arg2: tl.float32, arg3: tl.float64):
            x = tl.load(arg0)
            y = arg1 + arg2 + arg3
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.float64, device="cuda")

        args = (arg0, 1.0, 1.0, 1.0)

        compiled_kernel = floats[1,](*args)
        launcher = self._make_launcher(compiled_kernel)
        # TODO: in Pytorch's pinned version of triton, arg3 is typed as regular float
        # but in triton 3.3.0, this is fixed and it's 0ffd. We'll need to update later.
        self.assertEqual(launcher.arg_tys, "Offf")
        self.assertEqual(arg0, torch.tensor([3.0], dtype=torch.float64, device="cuda"))
        new_arg0 = torch.zeros(1, dtype=torch.float64, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(1, 1, 1, stream, new_arg0, 1.0, 1.0, 1.0)
        self.assertEqual(new_arg0, arg0)

    @skipIfRocm
    def test_basic_1arg(self):
        @triton.jit
        def simple_kernel_1_arg(arg0):
            x = tl.load(arg0)
            tl.store(arg0, x + 1)

        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        compiled_kernel = simple_kernel_1_arg[1,](arg0)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([1], dtype=torch.int32, device="cuda"))
        self.assertEqual(launcher.arg_tys, "O")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run(
            1,
            1,
            1,
            stream,
            new_arg0,
        )
        self.assertEqual(new_arg0, arg0)

    @skipIfRocm
    def test_constexpr(self):
        # Constexprs are compiled directly into the cubin file,
        # so we never need to pass it to StaticCudaLauncher.

        @triton.jit
        def kernel_constexpr(arg0, CONSTANT: tl.constexpr):
            x = tl.load(arg0)
            tl.store(arg0, x + CONSTANT)

        # Can't use make_launcher because constexpr needs to be constant
        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        compiled_kernel = kernel_constexpr[(1,)](arg0, CONSTANT=5)
        launcher = self._make_launcher(compiled_kernel)

        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device="cuda"))
        self.assertEqual(launcher.arg_tys, "O")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(
            1,
            1,
            1,
            stream,
            new_arg0,
        )
        self.assertEqual(new_arg0, arg0)

    @skipIfRocm
    def test_implied_constant(self):
        """xnumel is unused in this kernel, but isn't explicitly marked as a constexpr"""

        # This kernel was generated by inductor so it has a bunch of unused arguments. We don't change it
        @triton.jit
        def triton_red_fused_any_isinf_0(
            in_ptr0,
            out_ptr0,
            xnumel,  # noqa: F841
            r0_numel,
            XBLOCK: tl.constexpr,
            R0_BLOCK: tl.constexpr,
        ):
            xnumel = 1  # noqa: F841
            rnumel = r0_numel  # noqa: F841
            RBLOCK: tl.constexpr = R0_BLOCK  # noqa: F841
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # noqa: F841
            xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)  # noqa: F841
            r0_base = tl.arange(0, R0_BLOCK)[None, :]
            rbase = r0_base  # noqa: F841
            _tmp3 = tl.full([XBLOCK, R0_BLOCK], False, tl.int1)
            for r0_offset in range(0, r0_numel, R0_BLOCK):
                r0_index = r0_offset + r0_base
                r0_mask = r0_index < r0_numel
                roffset = r0_offset  # noqa: F841
                rindex = r0_index  # noqa: F841
                r0_0 = r0_index
                tmp0 = tl.load(
                    in_ptr0 + (r0_0), r0_mask, eviction_policy="evict_first", other=0.0
                )
                tmp1 = libdevice.isinf(tmp0).to(tl.int1)
                tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
                tmp4 = _tmp3 | tmp2
                _tmp3 = tl.where(r0_mask, tmp4, _tmp3)
            tmp3 = triton_helpers.any(_tmp3.to(tl.int8), 1)[:, None].to(tl.int1)
            tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp3, None)

        arg0 = torch.tensor([0.0, 0.5, float("inf"), 5], device="cuda")
        arg1 = torch.tensor([False], device="cuda")
        arg2 = torch.tensor([False], device="cuda")
        compiled_kernel = triton_red_fused_any_isinf_0[1,](
            arg0, arg1, 1, 128, XBLOCK=1, R0_BLOCK=1
        )
        launcher = self._make_launcher(compiled_kernel)

        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        # Don't pass in xnumel, as it is a constant
        launcher.run(1, 1, 1, stream, arg0, arg2, 128)
        self.assertEqual(arg1, arg2)

    @skipIfRocm
    def test_incompatible_args(self):
        # Just an easy way to test incompatible number of arguments
        @triton.jit
        def kernel_no_op():
            pass

        compiled_kernel = kernel_no_op[(1,)]()
        launcher = self._make_launcher(compiled_kernel)
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.run(1, 1, 1, stream)

    def test_slow_launch(self):
        @triton.jit
        def simple_kernel(arg0, arg1):
            x = tl.load(arg0)
            y = arg1
            tl.store(arg0, x + y)

        arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        arg1 = 5
        args = (arg0, arg1)
        compiled_kernel = simple_kernel[(1,)](*args)
        launcher = self._make_launcher(compiled_kernel)
        self.assertEqual(arg0, torch.tensor([5], dtype=torch.int32, device="cuda"))
        self.assertEqual(launcher.arg_tys, "Oi")
        new_arg0 = torch.zeros(1, dtype=torch.int32, device="cuda")
        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())
        launcher.slow_launch_kernel = True
        launcher.run(1, 1, 1, stream, new_arg0, arg1)
        self.assertEqual(new_arg0, arg0)

    @skipIfRocm
    def test_kernel_empty_tensor(self):
        # Triton kernel generated by torch.compile of the following:
        # @torch.compile()
        # def foo(x, y):
        #   return torch.cat(((x * 4), y + 10))

        # Running with example input:
        # torch._dynamo.decorators.mark_unbacked(t, 0)
        # x = torch.rand(0, device="cuda")
        # y = torch.rand(20, device="cuda")

        @triton.jit
        def triton_poi_fused_cat_0(
            in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK: tl.constexpr
        ):
            xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:].to(tl.int64)
            xmask = xindex < xnumel
            x0 = xindex
            tmp0 = x0
            tmp3 = ks0
            tmp4 = tmp0 < tmp3
            tmp5 = tl.load(
                in_ptr0 + (x0), xmask & tmp4, eviction_policy="evict_last", other=0.0
            )
            tmp6 = 4.0
            tmp7 = tmp5 * tmp6
            tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
            tmp9 = tl.where(tmp4, tmp7, tmp8)
            tmp10 = tmp0 >= tmp3
            tmp13 = tl.load(
                in_ptr1 + (x0 + ((-1) * ks0)),
                xmask & tmp10,
                eviction_policy="evict_last",
                other=0.0,
            )
            tmp14 = 10.0
            tmp15 = tmp13 + tmp14
            tmp16 = tl.full(tmp15.shape, 0.0, tmp15.dtype)
            tmp17 = tl.where(tmp10, tmp15, tmp16)
            tmp18 = tl.where(tmp4, tmp9, tmp17)
            tl.store(out_ptr0 + (x0), tmp18, xmask)

        arg0 = 0
        arg1 = torch.randn(0, device="cuda")
        arg2 = torch.randn(20, device="cuda")
        buf0 = torch.empty(20, device="cuda")
        buf1 = torch.empty(20, device="cuda")
        xnumel = 20 + arg0
        compiled_kernel = triton_poi_fused_cat_0[(1,)](
            arg1, arg2, buf0, arg0, xnumel, XBLOCK=32
        )
        launcher = self._make_launcher(compiled_kernel)

        device_interface = get_interface_for_device("cuda")
        stream = device_interface.get_raw_stream(device_interface.current_device())

        launcher.run(1, 1, 1, stream, arg1, arg2, buf1, arg0, xnumel)
        self.assertEqual(buf0, buf1)


@requires_cuda
@torch._inductor.config.patch(
    {"use_static_cuda_launcher": True, "strict_static_cuda_launcher": True}
)
class TestStaticTritonCompileResult(TestCase):
    """
    Tests static cuda launcher with torch.compile()
    """

    @skipIfRocm
    def test_basic_compile(self):
        @torch.compile
        def foo(x, y):
            return x + y

        x = torch.randn(10, device="cuda")
        y = torch.randn(10, device="cuda")
        self.assertEqual(foo(x, y), x + y)

    @skipIfRocm
    # The error gets raised on a worker, so we want to not use a separate process
    @torch._inductor.config.patch("compile_threads", 1)
    def test_incompatible_code(self):
        # User defined triton kernel
        @triton.jit
        def custom_kernel(arg_0, arg_1):
            x = tl.load(arg_0)
            y = arg_1
            tl.store(arg_0, x + y)

        @torch.compile
        def foo(x):
            custom_kernel[1,](x, 5)
            return x

        x = torch.randn(1, device="cuda")
        self.assertRaisesRegex(
            torch._inductor.exc.InductorError,
            "CannotStaticallyLaunchKernel: User defined triton kernel",
            lambda: foo(x),
        )

    @skipIfRocm
    def test_empty_tensor(self):
        @torch.compile()
        def foo(x, y):
            return torch.cat(((x * 4), y + 10))

        x = torch.rand(0, device="cuda")
        torch._dynamo.decorators.mark_unbacked(x, 0)
        y = torch.rand(20, device="cuda")
        result = foo(x, y)
        self.assertEqual(result, torch.cat(((x * 4), y + 10)))

    @skipIfRocm
    def test_any(self):
        def fn(x):
            return (
                x.any(-1),
                x.isinf().any(),
                torch.all(x.isinf(), dim=0),
                torch.all(torch.logical_not(x.isinf())),
            )

        compiled_fn = torch.compile(fn)
        arg = -torch.rand(64, device="cuda", dtype=torch.float64)
        eager_result = fn(arg)
        compiled_result = compiled_fn(arg)
        self.assertEqual(eager_result, compiled_result)
        arg[1] = float("inf")
        eager_result = fn(arg)
        compiled_result = compiled_fn(arg)
        self.assertEqual(eager_result, compiled_result)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
