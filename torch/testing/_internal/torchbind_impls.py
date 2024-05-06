import contextlib

import torch


_TORCHBIND_IMPLS_INITIALIZED = False


def init_torchbind_implementations():
    global _TORCHBIND_IMPLS_INITIALIZED
    if _TORCHBIND_IMPLS_INITIALIZED:
        return

    load_torchbind_test_lib()
    register_fake_operators()
    register_fake_classes()
    _TORCHBIND_IMPLS_INITIALIZED = True


# put these under a function because the corresponding library might not be loaded yet.
def register_fake_operators():
    @torch.library.register_fake("_TorchScriptTesting::takes_foo_python_meta")
    def fake_takes_foo(foo, z):
        return foo.add_tensor(z)

    @torch.library.register_fake("_TorchScriptTesting::queue_pop")
    def fake_queue_pop(tq):
        return tq.pop()

    @torch.library.register_fake("_TorchScriptTesting::queue_push")
    def fake_queue_push(tq, x):
        return tq.push(x)

    @torch.library.register_fake("_TorchScriptTesting::queue_size")
    def fake_queue_size(tq):
        return tq.size()

    def meta_takes_foo_list_return(foo, x):
        a = foo.add_tensor(x)
        b = foo.add_tensor(a)
        c = foo.add_tensor(b)
        return [a, b, c]

    def meta_takes_foo_tuple_return(foo, x):
        a = foo.add_tensor(x)
        b = foo.add_tensor(a)
        return (a, b)

    torch.ops._TorchScriptTesting.takes_foo_list_return.default.py_impl(
        torch._C.DispatchKey.Meta
    )(meta_takes_foo_list_return)

    torch.ops._TorchScriptTesting.takes_foo_tuple_return.default.py_impl(
        torch._C.DispatchKey.Meta
    )(meta_takes_foo_tuple_return)

    torch.ops._TorchScriptTesting.takes_foo.default.py_impl(torch._C.DispatchKey.Meta)(
        lambda cc, x: cc.add_tensor(x)
    )


def register_fake_classes():
    @torch._library.register_fake_class("_TorchScriptTesting::_Foo")
    class FakeFoo:
        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y

        @classmethod
        def from_real(cls, foo):
            (x, y), _ = foo.__getstate__()
            return cls(x, y)

        def add_tensor(self, z):
            return (self.x + self.y) * z

    @torch._library.register_fake_class("_TorchScriptTesting::_ContainsTensor")
    class FakeContainsTensor:
        def __init__(self, x: torch.Tensor):
            self.x = x

        @classmethod
        def from_real(cls, foo):
            ctx = torch.library.get_ctx()
            return cls(ctx.to_fake_tensor(foo.get()))

        def get(self):
            return self.x


def load_torchbind_test_lib():
    import unittest

    from torch.testing._internal.common_utils import (  # type: ignore[attr-defined]
        find_library_location,
        IS_FBCODE,
        IS_MACOS,
        IS_SANDCASTLE,
        IS_WINDOWS,
    )

    if IS_SANDCASTLE or IS_FBCODE:
        torch.ops.load_library("//caffe2/test/cpp/jit:test_custom_class_registrations")
    elif IS_MACOS:
        raise unittest.SkipTest("non-portable load_library call used in test")
    else:
        lib_file_path = find_library_location("libtorchbind_test.so")
        if IS_WINDOWS:
            lib_file_path = find_library_location("torchbind_test.dll")
        torch.ops.load_library(str(lib_file_path))


@contextlib.contextmanager
def _register_py_impl_temporarily(op_overload, key, fn):
    try:
        op_overload.py_impl(key)(fn)
        yield
    finally:
        del op_overload.py_kernels[key]
        op_overload._dispatch_cache.clear()
