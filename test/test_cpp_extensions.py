import os
import shutil
import sys
import unittest
import warnings
import numpy

import common_utils as common
import torch
import torch.backends.cudnn
import torch.utils.cpp_extension
from torch.utils.data import DataLoader
import torch.utils.data.chunk as chunk
from torch.utils.cpp_extension import CUDA_HOME


try:
    import torch_test_cpp_extension.cpp as cpp_extension
    import torch_test_cpp_extension.msnpu as msnpu_extension
except ImportError:
    warnings.warn(
        "test_cpp_extensions.py cannot be invoked directly. Run "
        "`python run_test.py -i cpp_extensions` instead."
    )


TEST_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
TEST_CUDNN = False
if TEST_CUDA:
    CUDNN_HEADER_EXISTS = os.path.isfile(os.path.join(CUDA_HOME, "include/cudnn.h"))
    TEST_CUDNN = (
        TEST_CUDA and CUDNN_HEADER_EXISTS and torch.backends.cudnn.is_available()
    )
IS_WINDOWS = sys.platform == "win32"


# This effectively allows re-using the same extension (compiled once) in
# multiple tests, just to split up the tested properties.
def dont_wipe_extensions_build_folder(func):
    func.dont_wipe = True
    return func


class TestCppExtension(common.TestCase):
    def setUp(self):
        test_name = self.id().split(".")[-1]
        dont_wipe = hasattr(getattr(self, test_name), "dont_wipe")
        if dont_wipe:
            print(
                "Test case {} has 'dont_wipe' attribute set, ".format(test_name) +
                "therefore not wiping extensions build folder before running the test"
            )
            return
        if sys.platform == "win32":
            print("Not wiping extensions build folder because Windows")
            return
        default_build_root = torch.utils.cpp_extension.get_default_build_root()
        if os.path.exists(default_build_root):
            shutil.rmtree(default_build_root)

    def test_extension_function(self):
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = cpp_extension.sigmoid_add(x, y)
        self.assertEqual(z, x.sigmoid() + y.sigmoid())

    def test_extension_module(self):
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4)
        expected = mm.get().mm(weights)
        result = mm.forward(weights)
        self.assertEqual(expected, result)

    def test_backward(self):
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, requires_grad=True)
        result = mm.forward(weights)
        result.sum().backward()
        tensor = mm.get()

        expected_weights_grad = tensor.t().mm(torch.ones([4, 4]))
        self.assertEqual(weights.grad, expected_weights_grad)

        expected_tensor_grad = torch.ones([4, 4]).mm(weights.t())
        self.assertEqual(tensor.grad, expected_tensor_grad)

    def test_jit_compile_extension(self):
        module = torch.utils.cpp_extension.load(
            name="jit_extension",
            sources=[
                "cpp_extensions/jit_extension.cpp",
                "cpp_extensions/jit_extension2.cpp",
            ],
            extra_include_paths=["cpp_extensions"],
            extra_cflags=["-g"],
            verbose=True,
        )
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

        # Checking we can call a method defined not in the main C++ file.
        z = module.exp_add(x, y)
        self.assertEqual(z, x.exp() + y.exp())

        # Checking we can use this JIT-compiled class.
        doubler = module.Doubler(2, 2)
        self.assertIsNone(doubler.get().grad)
        self.assertEqual(doubler.get().sum(), 4)
        self.assertEqual(doubler.forward().sum(), 8)

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cuda_extension(self):
        import torch_test_cpp_extension.cuda as cuda_extension

        x = torch.zeros(100, device="cuda", dtype=torch.float32)
        y = torch.zeros(100, device="cuda", dtype=torch.float32)

        z = cuda_extension.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_jit_cuda_extension(self):
        # NOTE: The name of the extension must equal the name of the module.
        module = torch.utils.cpp_extension.load(
            name="torch_test_cuda_extension",
            sources=[
                "cpp_extensions/cuda_extension.cpp",
                "cpp_extensions/cuda_extension.cu",
            ],
            extra_cuda_cflags=["-O2"],
            verbose=True,
        )

        x = torch.zeros(100, device="cuda", dtype=torch.float32)
        y = torch.zeros(100, device="cuda", dtype=torch.float32)

        z = module.sigmoid_add(x, y).cpu()

        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(not TEST_CUDNN, "CuDNN not found")
    def test_jit_cudnn_extension(self):
        # implementation of CuDNN ReLU
        if IS_WINDOWS:
            extra_ldflags = ["cudnn.lib"]
        else:
            extra_ldflags = ["-lcudnn"]
        module = torch.utils.cpp_extension.load(
            name="torch_test_cudnn_extension",
            sources=["cpp_extensions/cudnn_extension.cpp"],
            extra_ldflags=extra_ldflags,
            verbose=True,
            with_cuda=True,
        )

        x = torch.randn(100, device="cuda", dtype=torch.float32)
        y = torch.zeros(100, device="cuda", dtype=torch.float32)
        module.cudnn_relu(x, y)  # y=relu(x)
        self.assertEqual(torch.nn.functional.relu(x), y)
        with self.assertRaisesRegex(RuntimeError, "same size"):
            y_incorrect = torch.zeros(20, device="cuda", dtype=torch.float32)
            module.cudnn_relu(x, y_incorrect)

    def test_optional(self):
        has_value = cpp_extension.function_taking_optional(torch.ones(5))
        self.assertTrue(has_value)
        has_value = cpp_extension.function_taking_optional(None)
        self.assertFalse(has_value)

    def test_inline_jit_compile_extension_with_functions_as_list(self):
        cpp_source = """
        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {
          return x.tanh() + y.tanh();
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_with_functions_list",
            cpp_sources=cpp_source,
            functions="tanh_add",
            verbose=True,
        )

        self.assertEqual(module.tanh_add.__doc__.split("\n")[2], "tanh_add")

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.tanh_add(x, y)
        self.assertEqual(z, x.tanh() + y.tanh())

    def test_inline_jit_compile_extension_with_functions_as_dict(self):
        cpp_source = """
        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {
          return x.tanh() + y.tanh();
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_with_functions_dict",
            cpp_sources=cpp_source,
            functions={"tanh_add": "Tanh and then sum :D"},
            verbose=True,
        )

        self.assertEqual(module.tanh_add.__doc__.split("\n")[2], "Tanh and then sum :D")

    def test_inline_jit_compile_extension_multiple_sources_and_no_functions(self):
        cpp_source1 = """
        torch::Tensor sin_add(torch::Tensor x, torch::Tensor y) {
          return x.sin() + y.sin();
        }
        """

        cpp_source2 = """
        #include <torch/extension.h>
        torch::Tensor sin_add(torch::Tensor x, torch::Tensor y);
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
          m.def("sin_add", &sin_add, "sin(x) + sin(y)");
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension",
            cpp_sources=[cpp_source1, cpp_source2],
            verbose=True,
        )

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        z = module.sin_add(x, y)
        self.assertEqual(z, x.sin() + y.sin())

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_inline_jit_compile_extension_cuda(self):
        cuda_source = """
        __global__ void cos_add_kernel(
            const float* __restrict__ x,
            const float* __restrict__ y,
            float* __restrict__ output,
            const int size) {
          const auto index = blockIdx.x * blockDim.x + threadIdx.x;
          if (index < size) {
            output[index] = __cosf(x[index]) + __cosf(y[index]);
          }
        }

        torch::Tensor cos_add(torch::Tensor x, torch::Tensor y) {
          auto output = torch::zeros_like(x);
          const int threads = 1024;
          const int blocks = (output.numel() + threads - 1) / threads;
          cos_add_kernel<<<blocks, threads>>>(x.data<float>(), y.data<float>(), output.data<float>(), output.numel());
          return output;
        }
        """

        # Here, the C++ source need only declare the function signature.
        cpp_source = "torch::Tensor cos_add(torch::Tensor x, torch::Tensor y);"

        module = torch.utils.cpp_extension.load_inline(
            name="inline_jit_extension_cuda",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["cos_add"],
            verbose=True,
        )

        self.assertEqual(module.cos_add.__doc__.split("\n")[2], "cos_add")

        x = torch.randn(4, 4, device="cuda", dtype=torch.float32)
        y = torch.randn(4, 4, device="cuda", dtype=torch.float32)

        z = module.cos_add(x, y)
        self.assertEqual(z, x.cos() + y.cos())

    def test_inline_jit_compile_extension_throws_when_functions_is_bad(self):
        with self.assertRaises(ValueError):
            torch.utils.cpp_extension.load_inline(
                name="invalid_jit_extension", cpp_sources="", functions=5
            )

    def test_lenient_flag_handling_in_jit_extensions(self):
        cpp_source = """
        torch::Tensor tanh_add(torch::Tensor x, torch::Tensor y) {
          return x.tanh() + y.tanh();
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="lenient_flag_handling_extension",
            cpp_sources=cpp_source,
            functions="tanh_add",
            extra_cflags=["-g\n\n", "-O0 -Wall"],
            extra_include_paths=["       cpp_extensions\n"],
            verbose=True,
        )

        x = torch.zeros(100, dtype=torch.float32)
        y = torch.zeros(100, dtype=torch.float32)
        z = module.tanh_add(x, y).cpu()
        self.assertEqual(z, x.tanh() + y.tanh())

    def test_complex_registration(self):
        module = torch.utils.cpp_extension.load(
            name="complex_registration_extension",
            sources="cpp_extensions/complex_registration_extension.cpp",
            verbose=True,
        )

        # Make sure that the empty tensor is of the desired shape and type
        # Refer to https://github.com/pytorch/pytorch/issues/14829
        t = torch.empty(2, 2, dtype=torch.complex64)
        self.assertEqual(t.size(), torch.Size([2, 2]))
        self.assertEqual(t.type(), 'torch.ComplexFloatTensor')

    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_half_support(self):
        """
        Checks for an issue with operator< ambiguity for half when certain
        THC headers are included.

        See https://github.com/pytorch/pytorch/pull/10301#issuecomment-416773333
        for the corresponding issue.
        """
        cuda_source = """
        #include <THC/THCNumerics.cuh>

        template<typename T, typename U>
        __global__ void half_test_kernel(const T* input, U* output) {
            if (input[0] < input[1] || input[0] >= input[1]) {
                output[0] = 123;
            }
        }

        torch::Tensor half_test(torch::Tensor input) {
            auto output = torch::empty(1, input.options().dtype(torch::kFloat));
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "half_test", [&] {
                half_test_kernel<scalar_t><<<1, 1>>>(
                    input.data<scalar_t>(),
                    output.data<float>());
            });
            return output;
        }
        """

        module = torch.utils.cpp_extension.load_inline(
            name="half_test_extension",
            cpp_sources="torch::Tensor half_test(torch::Tensor input);",
            cuda_sources=cuda_source,
            functions=["half_test"],
            verbose=True,
        )

        x = torch.randn(3, device="cuda", dtype=torch.half)
        result = module.half_test(x)
        self.assertEqual(result[0], 123)

    def test_reload_jit_extension(self):
        def compile(code):
            return torch.utils.cpp_extension.load_inline(
                name="reloaded_jit_extension",
                cpp_sources=code,
                functions="f",
                verbose=True,
            )

        module = compile("int f() { return 123; }")
        self.assertEqual(module.f(), 123)

        module = compile("int f() { return 456; }")
        self.assertEqual(module.f(), 456)
        module = compile("int f() { return 456; }")
        self.assertEqual(module.f(), 456)

        module = compile("int f() { return 789; }")
        self.assertEqual(module.f(), 789)

    @dont_wipe_extensions_build_folder
    @common.skipIfRocm
    def test_cpp_frontend_module_has_same_output_as_python(self):
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        input = torch.randn(2, 5)
        cpp_linear = extension.Net(5, 2)
        cpp_linear.to(torch.float64)
        python_linear = torch.nn.Linear(5, 2)

        # First make sure they have the same parameters
        cpp_parameters = dict(cpp_linear.named_parameters())
        with torch.no_grad():
            python_linear.weight.copy_(cpp_parameters["fc.weight"])
            python_linear.bias.copy_(cpp_parameters["fc.bias"])

        cpp_output = cpp_linear.forward(input)
        python_output = python_linear(input)
        self.assertEqual(cpp_output, python_output)

        cpp_output.sum().backward()
        python_output.sum().backward()

        for p in cpp_linear.parameters():
            self.assertFalse(p.grad is None)

        self.assertEqual(cpp_parameters["fc.weight"].grad, python_linear.weight.grad)
        self.assertEqual(cpp_parameters["fc.bias"].grad, python_linear.bias.grad)

    @dont_wipe_extensions_build_folder
    @common.skipIfRocm
    def test_cpp_frontend_module_python_inter_op(self):
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        # Create a torch.nn.Module which uses the C++ module as a submodule.
        class M(torch.nn.Module):
            def __init__(self):
                super(M, self).__init__()
                self.x = torch.nn.Parameter(torch.tensor(1.0))
                self.net = extension.Net(3, 5)

            def forward(self, input):
                return self.net.forward(input) + self.x

        net = extension.Net(5, 2)
        net.double()
        net.to(torch.get_default_dtype())
        self.assertEqual(str(net), "Net")

        # Further embed the torch.nn.Module into a Sequential, and also add the
        # C++ module as an element of the Sequential.
        sequential = torch.nn.Sequential(M(), torch.nn.Tanh(), net, torch.nn.Sigmoid())

        input = torch.randn(2, 3)
        # Try calling the module!
        output = sequential.forward(input)
        # The call operator is bound to forward too.
        self.assertEqual(output, sequential(input))
        self.assertEqual(list(output.shape), [2, 2])

        # Do changes on the module hierarchy.
        old_dtype = torch.get_default_dtype()
        sequential.to(torch.float64)
        sequential.to(torch.float32)
        sequential.to(old_dtype)
        self.assertEqual(sequential[2].parameters()[0].dtype, old_dtype)

        # Make sure we can access these methods recursively.
        self.assertEqual(len(list(sequential.parameters())), len(net.parameters()) * 2 + 1)
        self.assertEqual(len(list(sequential.named_parameters())), len(net.named_parameters()) * 2 + 1)
        self.assertEqual(len(list(sequential.buffers())), len(net.buffers()) * 2)
        self.assertEqual(len(list(sequential.modules())), 8)

        # Test clone()
        net2 = net.clone()
        self.assertEqual(len(net.parameters()), len(net2.parameters()))
        self.assertEqual(len(net.buffers()), len(net2.buffers()))
        self.assertEqual(len(net.modules()), len(net2.modules()))

        # Try differentiating through the whole module.
        for parameter in net.parameters():
            self.assertIsNone(parameter.grad)
        output.sum().backward()
        for parameter in net.parameters():
            self.assertFalse(parameter.grad is None)
            self.assertGreater(parameter.grad.sum(), 0)

        # Try calling zero_grad()
        net.zero_grad()
        for p in net.parameters():
            self.assertEqual(p.grad, torch.zeros_like(p))

        # Test train(), eval(), training (a property)
        self.assertTrue(net.training)
        net.eval()
        self.assertFalse(net.training)
        net.train()
        self.assertTrue(net.training)
        net.eval()

        # Try calling the additional methods we registered.
        biased_input = torch.randn(4, 5)
        output_before = net.forward(biased_input)
        bias = net.get_bias().clone()
        self.assertEqual(list(bias.shape), [2])
        net.set_bias(bias + 1)
        self.assertEqual(net.get_bias(), bias + 1)
        output_after = net.forward(biased_input)

        self.assertNotEqual(output_before, output_after)

        # Try accessing parameters
        self.assertEqual(len(net.parameters()), 2)
        np = net.named_parameters()
        self.assertEqual(len(np), 2)
        self.assertIn("fc.weight", np)
        self.assertIn("fc.bias", np)

        self.assertEqual(len(net.buffers()), 1)
        nb = net.named_buffers()
        self.assertEqual(len(nb), 1)
        self.assertIn("buf", nb)
        self.assertEqual(nb[0][1], torch.eye(5))

    @dont_wipe_extensions_build_folder
    @common.skipIfRocm
    def test_cpp_frontend_module_has_up_to_date_attributes(self):
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        net = extension.Net(5, 2)

        self.assertEqual(len(net._parameters), 0)
        net.add_new_parameter("foo", torch.eye(5))
        self.assertEqual(len(net._parameters), 1)

        self.assertEqual(len(net._buffers), 1)
        net.add_new_buffer("bar", torch.eye(5))
        self.assertEqual(len(net._buffers), 2)

        self.assertEqual(len(net._modules), 1)
        net.add_new_submodule("fc2")
        self.assertEqual(len(net._modules), 2)

    @dont_wipe_extensions_build_folder
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    @common.skipIfRocm
    def test_cpp_frontend_module_python_inter_op_with_cuda(self):
        extension = torch.utils.cpp_extension.load(
            name="cpp_frontend_extension",
            sources="cpp_extensions/cpp_frontend_extension.cpp",
            verbose=True,
        )

        net = extension.Net(5, 2)
        for p in net.parameters():
            self.assertTrue(p.device.type == "cpu")
        cpu_parameters = [p.clone() for p in net.parameters()]

        device = torch.device("cuda", 0)
        net.to(device)

        for i, p in enumerate(net.parameters()):
            self.assertTrue(p.device.type == "cuda")
            self.assertTrue(p.device.index == 0)
            self.assertEqual(cpu_parameters[i], p)

        net.cpu()
        net.add_new_parameter("a", torch.eye(5))
        net.add_new_parameter("b", torch.eye(5))
        net.add_new_buffer("c", torch.eye(5))
        net.add_new_buffer("d", torch.eye(5))
        net.add_new_submodule("fc2")
        net.add_new_submodule("fc3")

        for p in net.parameters():
            self.assertTrue(p.device.type == "cpu")

        net.cuda()

        for p in net.parameters():
            self.assertTrue(p.device.type == "cuda")

    def test_returns_shared_library_path_when_is_python_module_is_true(self):
        source = """
        #include <torch/script.h>
        torch::Tensor func(torch::Tensor x) { return x; }
        static torch::jit::RegisterOperators r("test::func", &func);
        """
        torch.utils.cpp_extension.load_inline(
            name="is_python_module",
            cpp_sources=source,
            functions="func",
            verbose=True,
            is_python_module=False,
        )
        self.assertEqual(torch.ops.test.func(torch.eye(5)), torch.eye(5))

    @unittest.skipIf(IS_WINDOWS, "Not available on Windows")
    def test_no_python_abi_suffix_sets_the_correct_library_name(self):
        # For this test, run_test.py will call `python setup.py install` in the
        # cpp_extensions/no_python_abi_suffix_test folder, where the
        # `BuildExtension` class has a `no_python_abi_suffix` option set to
        # `True`. This *should* mean that on Python 3, the produced shared
        # library does not have an ABI suffix like
        # "cpython-37m-x86_64-linux-gnu" before the library suffix, e.g. "so".
        # On Python 2 there is no ABI suffix anyway.
        root = os.path.join("cpp_extensions", "no_python_abi_suffix_test", "build")
        matches = [f for _, _, fs in os.walk(root) for f in fs if f.endswith("so")]
        self.assertEqual(len(matches), 1, str(matches))
        self.assertEqual(matches[0], "no_python_abi_suffix_test.so", str(matches))

    def test_set_default_type_also_changes_aten_default_type(self):
        module = torch.utils.cpp_extension.load_inline(
            name="test_set_default_type",
            cpp_sources="torch::Tensor get() { return torch::empty({}); }",
            functions="get",
            verbose=True,
        )

        initial_default = torch.get_default_dtype()
        try:
            self.assertEqual(module.get().dtype, initial_default)
            torch.set_default_dtype(torch.float64)
            self.assertEqual(module.get().dtype, torch.float64)
            torch.set_default_dtype(torch.float32)
            self.assertEqual(module.get().dtype, torch.float32)
            torch.set_default_dtype(torch.float16)
            self.assertEqual(module.get().dtype, torch.float16)
        finally:
            torch.set_default_dtype(initial_default)


    def test_compilation_error_formatting(self):
        # Test that the missing-semicolon error message has linebreaks in it. 
        # This'll fail if the message has been munged into a single line.
        # It's hard to write anything more specific as every compiler has it's own
        # error formatting.
        with self.assertRaises(RuntimeError) as e:
            torch.utils.cpp_extension.load_inline(
                name="test_compilation_error_formatting",
                cpp_sources="int main() { return 0 }") 
        pattern = r'.*(\\n|\\r).*'
        self.assertNotRegex(str(e), pattern)

    def test_dummy_chunkdataset_bindings(self):

        """
        This class serves as a basic example on how to use ChunkDataset API python bindings

        The main steps for a ChunkDataset implementation with built-in type Example are:
        C++ Steps: Check `test/cpp_extensions/extension.cpp`:
            Step 1: [optional] Make `std::vector<BUILT_IN_TYPE>` opaque to Python through `PYBIND11_MAKE_OPAQUE()`
                        Opaque types prevent memory copy between C++ and Python

        Python Steps: Check `test_dummy_chunkdataset_bindings` below:
            Step 1: Instantiate a `Sampler` for the chunk sampler
            Step 2: Instantiate a `SamplerWrapper` for the chunk sampler in the previous step
            Step 3: Instantiate a `Sampler` for the example sampler
            Step 4: Instantiate a `SamplerWrapper` for the example sampler in the previous step
            Step 5: Instantiate a specific `ChunkDataReader`
            Step 6: Instantiate a `ChunkDatasetOptions`
            Step 7: Instantiate a `ChunkDataset`
            Step 8: Instantiate a `ChunkDatasetWrapper`
            Step 9: Instantiate a `DataLoader`
            Step 10: Iterate on `DataLoader`

        The API is almost identical to the C++ version. The difference is the need for the
        SampleWrapper on Python implementation that is not required by C++ counterpart.
        This is due to the fact that Python DataLoader uses multiprocessing module for
        parallelism as opposed to multi-threading
        """
        chunk_count = 3
        batch_size = 5
        cache_size = 100
        preloaders = 1
        chunk_sampler = chunk.SequentialSampler(size=chunk_count)
        example_sampler = chunk.SequentialSampler(size=batch_size)
        chunk_sampler_wrapper = chunk.SamplerWrapper(sampler=chunk_sampler)
        example_sampler_wrapper = chunk.SamplerWrapper(sampler=example_sampler)
        reader = cpp_extension.DummyChunkDataReader()
        opt = chunk.ChunkDatasetOptions(preloader_count=preloaders, batch_size=batch_size, cache_size=cache_size)

        dummy_chunkdataset = cpp_extension.DummyChunkDataset(chunk_reader=reader,
                                                             chunk_sampler=chunk_sampler_wrapper,
                                                             example_sampler=example_sampler_wrapper,
                                                             options=opt)

        trainset = chunk.ChunkDatasetWrapper(dummy_chunkdataset)
        trainset.reset()
        trainloader = DataLoader(dataset=trainset,
                                 batch_size=None,
                                 pin_memory=True)
        for i, actual in enumerate(trainloader, 0):
            expected = [torch.tensor([j], dtype=torch.long)
                        for j in list(range(batch_size * i, batch_size * i + batch_size))]
            self.assertEqual(expected, actual)

    def test_foo_chunkdataset_bindings(self):
        """
        This class serves as a more complex example on how to use ChunkDataset API python bindings

        The main steps for a full ChunkDataset implementation are:

        C++ Steps: Check `test/cpp_extensions/extension.cpp`:
            Step 1: Define `FooExampleType` struct (aka ExampleType)
            Step 2: Define `FooChunkDataReader` class by extending `torch::data::datasets::ChunkDataReader<FooExampleType>`
            Step 3: Bind `FooExampleType` struct
            Step 4: [optional] Make `std::vector<FooExampleType>` opaque to Python through `PYBIND11_MAKE_OPAQUE()`
                        Opaque types prevent memory copy between C++ and Python
            Step 4.1: Bind `std::vector<FooExampleType>` through `py::bind_vector<FooExampleType>()` (aka BatchType)
            Step 6: Bind `FooChunkDataReader` binding through `py::class_<FooChunkDataReader>`
            Step 7: Bind `FooChunkDataset` binding through `py::class_<FooChunkDataReader>`

        Python Steps: Check `test_foo_chunkdataset_bindings` below:
            Step 1: Instantiate a `Sampler` for the chunk sampler
            Step 2: Instantiate a `SamplerWrapper` for the chunk sampler in the previous step
            Step 3: Instantiate a `Sampler` for the example sampler
            Step 4: Instantiate a `SamplerWrapper` for the example sampler in the previous step
            Step 5: Instantiate a specific `ChunkDataReader`
            Step 6: Instantiate a `ChunkDatasetOptions`
            Step 7: Instantiate a `ChunkDataset`
            Step 8: Instantiate a `ChunkDatasetWrapper`
            Step 9: Instantiate a `DataLoader`
            Step 9.1: [optional] If BatchType doesn't contain tensors, numpy arrays, numbers, dicts or lists, implement `collate_fn`
            Step 10: Iterate on `DataLoader`

        The API is almost identical to the C++ version. The difference is the need for the
        SampleWrapper on Python implementation that is not required by C++ counterpart.
        This is due to the fact that Python DataLoader uses multiprocessing module for
        parallelism as opposed to multi-threading
        """

        def collate_fn(batch):
            if batch is not None:
                # Output is a dictionary
                dict = {}

                # Utils to allocate shared memory for tensors
                # We need to know the exact amount of memory to preallocate it
                # in shared memory so the main process can access it quickly
                # Send each tensor individually to shared memory leads to 2x perf hit
                features = []
                features_numel = 0
                features_out = None
                labels = []
                labels_numel = 0
                labels_out = None

                for u in batch:
                    # C++ FooExampleType to Pytorch Tensors (zero-copy)
                    feature_tensor = torch.from_numpy(numpy.array(u.feature_))
                    label_tensor = torch.from_numpy(numpy.array(u.label_))

                    # Determining tensor size for using shared memory
                    features_numel += feature_tensor.numel()
                    features.append(feature_tensor)
                    labels_numel += label_tensor.numel()
                    labels.append(label_tensor)

                # Allocating shared memory
                features_shared_storage = features[0].storage(
                )._new_shared(features_numel)
                features_out = features[0].new(features_shared_storage)
                labels_shared_storage = labels[0].storage()._new_shared(labels_numel)
                labels_out = labels[0].new(labels_shared_storage)

                # Stacking (copying) tensors into shared memory
                torch.stack(features, out=features_out)
                torch.stack(labels, out=labels_out)

                # Return shared memory on torch.multprocessing queues (zero-copy)
                dict['feature'] = features_out
                dict['label'] = labels_out
                return dict

        def worker_init_fn(worker_id):
            # A recent change on pytorch enabled multithreading by default
            # Dataloader logic requires a single thread, though
            # Until the https://github.com/pytorch/pytorch/issues/19213 is resolved,
            # you have to create an environment variable OMP_NUM_THREADS=1 as a workaround
            torch.set_num_threads(1)
            print('Initializing worker {} with {} threads'.format(
                worker_id, torch.get_num_threads()))
            dataset = torch.utils.data.get_worker_info().dataset
            chunk_sampler = dataset.chunk_sampler()
            chunk_sampler.set_current_stride(stride=worker_id)
            dataset.reset()

        chunk_count = 1
        batch_size = 5
        cache_size = 100
        preloaders = 1
        num_workers = 1
        chunk_sampler = chunk.SequentialSampler(size=chunk_count)
        example_sampler = chunk.SequentialSampler(size=batch_size)
        chunk_sampler_wrapper = chunk.SamplerWrapper(sampler=chunk_sampler)
        example_sampler_wrapper = chunk.SamplerWrapper(sampler=example_sampler)
        reader = cpp_extension.FooChunkDataReader()
        opt = chunk.ChunkDatasetOptions(preloader_count=preloaders, batch_size=batch_size, cache_size=cache_size)

        foo_chunkdataset = cpp_extension.FooChunkDataset(chunk_reader=reader,
                                                         chunk_sampler=chunk_sampler_wrapper,
                                                         example_sampler=example_sampler_wrapper,
                                                         options=opt)

        trainset = chunk.ChunkDatasetWrapper(foo_chunkdataset)
        trainset.reset()
        trainloader = DataLoader(dataset=trainset,
                                 num_workers=num_workers,
                                 batch_size=None,
                                 pin_memory=True,
                                 collate_fn=collate_fn,
                                 worker_init_fn=worker_init_fn)
        for i, actual in enumerate(trainloader, 0):
            expected = {'feature': torch.tensor([j for j in list(range(batch_size * i, batch_size * i + batch_size))]),
                        'label': torch.tensor([j for j in list(range(batch_size * i + 1, batch_size * i + 1 + batch_size))])}
            self.assertEqual(expected, actual)

class TestMSNPUTensor(common.TestCase):
    @classmethod
    def setUpClass(cls):
        msnpu_extension.init_msnpu_extension()

    def test_unregistered(self):
        a = torch.empty(5, 5, device='cpu')
        with self.assertRaisesRegex(RuntimeError, "No function registered"):
            b = torch.empty(5, 5, device='msnpu')

    def test_zeros(self):
        a = torch.zeros(5, 5, device='cpu')
        self.assertEqual(a.device, torch.device('cpu'))
        self.assertEqual(a.sum(), 0)

        b = torch.zeros(5, 5, device='msnpu')
        self.assertEqual(b.device, torch.device('msnpu', 0))
        self.assertEqual(msnpu_extension.get_test_int(), 0)
        self.assertEqual(torch.get_default_dtype(), b.dtype)

        c = torch.zeros((5, 5), dtype=torch.int64, device='msnpu')
        self.assertEqual(msnpu_extension.get_test_int(), 0)
        self.assertEqual(torch.int64, c.dtype)

    def test_add(self):
        a = torch.zeros(5, 5, device='msnpu')
        self.assertEqual(msnpu_extension.get_test_int(), 0)

        b = torch.zeros(5, 5, device='msnpu')
        self.assertEqual(msnpu_extension.get_test_int(), 0)

        c = torch.add(a, b)
        self.assertEqual(msnpu_extension.get_test_int(), 1)

    def test_backwards(self):
        a = torch.zeros(5, 5, device='msnpu', requires_grad=True)
        self.assertEqual(msnpu_extension.get_test_int(), 0)

        b = torch.zeros(5, 5, device='msnpu')
        self.assertEqual(msnpu_extension.get_test_int(), 0)

        c = torch.kl_div(a, b)
        self.assertEqual(msnpu_extension.get_test_int(), 3)

        d = c.sum()
        self.assertEqual(msnpu_extension.get_test_int(), 2)

        d.backward(torch.zeros(0, device='msnpu'))
        self.assertEqual(msnpu_extension.get_test_int(), 4)


if __name__ == "__main__":
    common.run_tests()
