# Owner(s): ["module: functorch"]
import json
import tempfile
import zipfile
from pathlib import Path

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
from torch._higher_order_ops.torchbind import CallTorchBind, enable_torchbind_tracing
from torch._inductor import aot_compile, ir
from torch._inductor.package import package_aoti
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.torchbind_impls import (
    _empty_tensor_queue,
    init_torchbind_implementations,
)


class TestTorchbind(TestCase):
    def setUp(self):
        super().setUp()
        init_torchbind_implementations()

    def get_exported_model(self):
        """
        Returns the ExportedProgram, example inputs, and result from calling the
        eager model with those inputs
        """

        class M(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
                self.b = torch.randn(2, 3)

            def forward(self, x):
                x = x + self.b
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                c = self.attr.add_tensor(x)
                return x + b + c

        m = M()
        inputs = (torch.ones(2, 3),)
        orig_res = m(*inputs)

        # We can't directly torch.compile because dynamo doesn't trace ScriptObjects yet
        with enable_torchbind_tracing():
            ep = torch.export.export(m, inputs, strict=False)

        return ep, inputs, orig_res, m

    def test_torchbind_inductor(self):
        ep, inputs, orig_res, _ = self.get_exported_model()
        compiled = torch._inductor.compile(ep.module(), inputs)

        new_res = compiled(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_compile(self):
        _, inputs, orig_res, mod = self.get_exported_model()
        new_res = torch.compile(mod, backend="inductor")(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res))

    def test_torchbind_get_buf_bytes(self):
        a = torch.classes._TorchScriptTesting._Foo(10, 20)
        buffer = ir.TorchBindObject(name="a", value=a)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 0)

        t = torch.randn(2, 3)
        b = torch.classes._TorchScriptTesting._ContainsTensor(t)
        buffer = ir.TorchBindObject(name="b", value=b)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 2 * 3 * 4)

        q = _empty_tensor_queue()
        buffer = ir.TorchBindObject(name="q", value=q)
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 0)

        q.push(torch.ones(2, 3))
        size = buffer.get_buf_bytes()
        self.assertEqual(size, 2 * 3 * 4)

    def test_torchbind_hop_schema(self):
        foo = torch.classes._TorchScriptTesting._Foo(10, 20)
        foo_ir = ir.TorchBindObject(name="foo", value=foo)
        schema = CallTorchBind.schema(foo_ir, "add")
        self.assertEqual(
            str(schema),
            "call_torchbind(__torch__.torch.classes._TorchScriptTesting._Foo obj, str method, int _1) -> int _0",
        )

    def test_torchbind_aot_compile(self):
        ep, inputs, _, _ = self.get_exported_model()
        aoti_files = aot_compile(
            ep.module(), inputs, options={"aot_inductor.package": True}
        )

        custom_objs_config = None
        custom_obj_0 = None
        extern_json = None
        for file in aoti_files:
            if file.endswith("/custom_objs_config.json"):
                custom_objs_config = file
            elif file.endswith("/custom_obj_0"):
                custom_obj_0 = file
            elif file.endswith(".json") and "metadata" not in file:
                extern_json = file

        self.assertIsNotNone(custom_objs_config)
        self.assertIsNotNone(custom_obj_0)
        self.assertIsNotNone(extern_json)

        with open(custom_objs_config) as file:
            data = json.load(file)
            self.assertEqual(data, {"_torchbind_obj0": "custom_obj_0"})

        with open(extern_json) as file:
            data = json.load(file)
            self.assertEqual(
                data,
                {
                    "nodes": [
                        {
                            "name": "buf3",
                            "node": {
                                "target": "_TorchScriptTesting::takes_foo_tuple_return",
                                "inputs": [
                                    {
                                        "name": "foo",
                                        "arg": {
                                            "as_custom_obj": {
                                                "name": "_torchbind_obj0",
                                                "class_fqn": "__torch__.torch.classes._TorchScriptTesting._Foo",
                                            }
                                        },
                                        "kind": 1,
                                    },
                                    {
                                        "name": "x",
                                        "arg": {"as_tensor": {"name": "buf2"}},
                                        "kind": 1,
                                    },
                                ],
                                "outputs": [
                                    {"as_tensor": {"name": "buf4"}},
                                    {"as_tensor": {"name": "buf5"}},
                                ],
                                "metadata": {},
                                "is_hop_single_tensor_return": None,
                            },
                        },
                        {
                            "name": "buf7",
                            "node": {
                                "target": "_TorchScriptTesting::takes_foo",
                                "inputs": [
                                    {
                                        "name": "foo",
                                        "arg": {
                                            "as_custom_obj": {
                                                "name": "_torchbind_obj0",
                                                "class_fqn": "__torch__.torch.classes._TorchScriptTesting._Foo",
                                            }
                                        },
                                        "kind": 1,
                                    },
                                    {
                                        "name": "x",
                                        "arg": {"as_tensor": {"name": "buf6"}},
                                        "kind": 1,
                                    },
                                ],
                                "outputs": [{"as_tensor": {"name": "buf8"}}],
                                "metadata": {},
                                "is_hop_single_tensor_return": None,
                            },
                        },
                        {
                            "name": "buf9",
                            "node": {
                                "target": "call_torchbind",
                                "inputs": [
                                    {
                                        "name": "obj",
                                        "arg": {
                                            "as_custom_obj": {
                                                "name": "_torchbind_obj0",
                                                "class_fqn": "__torch__.torch.classes._TorchScriptTesting._Foo",
                                            }
                                        },
                                        "kind": 1,
                                    },
                                    {
                                        "name": "method",
                                        "arg": {"as_string": "add_tensor"},
                                        "kind": 1,
                                    },
                                    {
                                        "name": "_1",
                                        "arg": {"as_tensor": {"name": "buf2"}},
                                        "kind": 1,
                                    },
                                ],
                                "outputs": [{"as_tensor": {"name": "buf10"}}],
                                "metadata": {},
                                "is_hop_single_tensor_return": None,
                            },
                        },
                    ]
                },
            )

        # Test that the files are packaged
        with tempfile.NamedTemporaryFile(suffix=".pt2") as f:
            package_path = package_aoti(f.name, aoti_files)

            with tempfile.TemporaryDirectory() as tmp_dir, zipfile.ZipFile(
                package_path, "r"
            ) as zip_ref:
                zip_ref.extractall(tmp_dir)
                tmp_path_model = Path(tmp_dir) / "data" / "aotinductor" / "model"
                tmp_path_constants = Path(tmp_dir) / "data" / "constants"

                self.assertTrue((tmp_path_model / "custom_objs_config.json").exists())
                self.assertTrue((tmp_path_constants / "custom_obj_0").exists())

        # TODO: add accuracy test after we support loading and running compiled models with
        # torchbind objects.

    @torch._inductor.config.patch("aot_inductor.use_runtime_constant_folding", True)
    def test_torchbind_aot_compile_constant_folding(self):
        ep, inputs, _, _ = self.get_exported_model()
        aot_compile(ep.module(), inputs, options={"aot_inductor.package": True})
        # TODO: add accuracy test after we support loading and running compiled models with
        # torchbind objects.


if __name__ == "__main__":
    run_tests()
