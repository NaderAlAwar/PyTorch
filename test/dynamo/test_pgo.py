# Owner(s): ["module: dynamo"]

import contextlib
import importlib
import os
import sys

import torch._dynamo.config
import torch._dynamo.test_case
import torch.compiler.config
from torch._dynamo.testing import CompileCounter
from torch._inductor.utils import clear_inductor_caches, fresh_inductor_cache


# LOL.  https://github.com/pytorch/pytorch/issues/139252
spec = importlib.util.spec_from_file_location(
    "mock_cache", os.path.join(os.path.dirname(__file__), "../inductor/mock_cache.py")
)
mock_cache = importlib.util.module_from_spec(spec)
sys.modules["mock_cache"] = mock_cache
spec.loader.exec_module(mock_cache)


class PgoTest(torch._dynamo.test_case.TestCase):
    def setUp(self):
        super().setUp()
        self._test_stack = contextlib.ExitStack()
        self._test_stack.enter_context(torch.compiler.config.patch(job_id=self.id()))
        self._test_stack.enter_context(
            torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
        )
        if os.environ.get("INDUCTOR_TEST_DISABLE_FRESH_CACHE") != "1":
            self._test_stack.enter_context(fresh_inductor_cache())
        mock_cache.PatchCaches.setUp()

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()
        self._test_stack.close()
        mock_cache.PatchCaches.tearDown()

    def reset(self):
        torch._dynamo.reset()
        clear_inductor_caches()

    def test_basic(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        f(torch.randn(2, 3))
        f(torch.randn(2, 4))
        self.assertEqual(cnts.frame_count, 2)

        self.reset()
        cnts.clear()

        f(torch.randn(2, 5))
        f(torch.randn(2, 6))
        self.assertEqual(cnts.frame_count, 1)

    def test_distinct_compile_id(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        with torch.compiler.config.patch(job_id="foo"):
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
        self.assertEqual(cnts.frame_count, 2)

        self.reset()
        cnts.clear()

        with torch.compiler.config.patch(job_id="bar"):
            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
        self.assertEqual(cnts.frame_count, 2)

        torch._dynamo.reset()
        clear_inductor_caches()
        cnts.clear()

        with torch.compiler.config.patch(job_id="foo"):
            f(torch.randn(2, 7))
            f(torch.randn(2, 8))
        self.assertEqual(cnts.frame_count, 1)

    # TODO: to test local need to ensure the local filesystem gets cleared out
    @torch._dynamo.config.patch(
        automatic_dynamic_remote_pgo=True, automatic_dynamic_local_pgo=False
    )
    def test_remote_basic(self):
        cnts = CompileCounter()

        @torch.compile(backend=cnts, fullgraph=True)
        def f(x):
            return x * 2

        with mock_cache.PatchCaches():
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(
                mock_cache.global_stats.dynamo_pgo, mock_cache.Stats(2, 0, 1)
            )

            self.reset()
            cnts.clear()

            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(
                mock_cache.global_stats.dynamo_pgo, mock_cache.Stats(2, 1, 1)
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
