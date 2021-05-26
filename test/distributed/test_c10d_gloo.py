import copy
import logging
import math
import operator
import os
import random
import sys
import tempfile
import unittest
from functools import reduce
from itertools import groupby

import torch
import torch.distributed as c10d

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_gloo,
    skip_if_lt_x_gpu,
    simple_sparse_reduce_tests,
    skip_if_win32,
    create_device,
    with_dist_debug_levels,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    retry_on_connect_failures,
    TEST_WITH_TSAN,
)
import test_c10d_common
from test_c10d_common import (
    LOOPBACK,
    gpus_for_rank,
    Task,
    ModuleForDdpCommHook,
    SparseGradientModule,
    AbstractProcessGroupWrapperTest,
)


def simple_reduce_tests(rank, world_size):
    tests = [
        (
            c10d.ReduceOp.SUM,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.tensor([rank + 1.0]),
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.tensor([rank + 1.0]),
            torch.tensor([world_size]),
        ),
    ]

    # Generate tests for BAND.
    # The bit that is set changes in every iteration to check
    # that the output changes accordingly.
    for i in range(4):
        vin = rank | (1 << i)
        vout = 1 << i
        tests.append(
            (
                c10d.ReduceOp.BAND,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for BOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-OR'ed.
    for i in range(1, 5):
        vin = reduce(operator.or_, [rank * i + j for j in range(i)])
        vout = reduce(operator.or_, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for XOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-XOR'ed.
    for i in range(1, 5):
        vin = reduce(operator.xor, [rank * i + j for j in range(i)])
        vout = reduce(operator.xor, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BXOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    return tests


def simple_coalesced_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([rank + 1]), torch.tensor([(rank + 1) ** 2])],
            [
                torch.tensor([float(world_size * (world_size + 1) / 2)]),
                torch.tensor(
                    [float(world_size * (world_size + 1) * (2 * world_size + 1) / 6)]
                ),
            ],
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([rank + 1.0]), torch.tensor([rank + 2.0])],
            [
                torch.tensor([float(math.factorial(world_size))]),
                torch.tensor([float(math.factorial(world_size + 1))]),
            ],
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([rank + x]) for x in [0.0, 1.0]],
            [torch.tensor([0.0]), torch.tensor([1.0])],
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([rank + x]) for x in [1.0, 2.0]],
            [torch.tensor([world_size]), torch.tensor([world_size + 1.0])],
        ),
    ]


def simple_multi_input_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([2 * rank + 0.0]), torch.tensor([2 * rank + 1.0])],
            torch.tensor([float(world_size * (2 * world_size - 1))]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([float(math.factorial(2 * world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([2 * world_size]),
        ),
    ]


class RendezvousEnvTest(TestCase):
    @requires_gloo()
    @retry_on_connect_failures
    def test_logging_init(self):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"

        previous_handlers = logging.root.handlers

        c10d.init_process_group(backend="gloo", init_method="env://")

        current_handlers = logging.root.handlers
        self.assertEqual(len(previous_handlers), len(current_handlers))
        for current, previous in zip(current_handlers, previous_handlers):
            self.assertEqual(current, previous)

        c10d.destroy_process_group()


class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    @requires_gloo()
    @retry_on_connect_failures
    def test_default_store_timeout_gloo(self):
        self._test_default_store_timeout("gloo")

@requires_gloo()
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class ProcessGroupGlooWrapperTest(AbstractProcessGroupWrapperTest):
    def setUp(self):
        super(ProcessGroupGlooWrapperTest, self).setUp()

    def opts(self, threads=2, timeout=10.0):
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = timeout
        opts._devices = [create_device(interface=LOOPBACK)]
        opts._threads = threads
        return opts

    def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )
        if with_new_group:
            pg = c10d.new_group(backend="gloo")
        else:
            _pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts(timeout=timeout))
            pg = c10d._create_process_group_wrapper(
                _pg,
                "unused",
                store,
                self.rank,
                self.world_size,
                timeout=timeout,
            )
        return pg

    def test_collective_hang(self):
        pg = self._create_wrapper_pg(timeout=2.0)
        self._test_collective_hang(pg)

    # NOTE: these tests are separated by debug level instead of combined into
    # one due to https://github.com/pytorch/pytorch/issues/55967, they can be
    # combined after that is resolved.
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg)

    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg)

    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg)

    @with_dist_debug_levels(levels=["OFF"])
    def test_collective_shape_mismatch(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_cuda_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch_cuda(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_cuda_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["OFF"])
    def test_collective_shape_mismatch_cuda(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg, use_cuda=True)

@requires_gloo()
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class ProcessGroupGlooTest(MultiProcessTestCase):
    def setUp(self):
        super(ProcessGroupGlooTest, self).setUp()

        # For Windows platform, Python does not support fork, change it to spawn here.
        if sys.platform == "win32":
            self._spawn_processes()
        else:
            self._fork_processes()

    def opts(self, threads=2):
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 5.0
        opts._devices = [create_device(interface=LOOPBACK)]
        opts._threads = threads
        return opts

    def test_multi_device_constructor(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 5.0
        opts._devices = [
            create_device(interface=LOOPBACK),
            create_device(interface=LOOPBACK),
        ]
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, opts)

        # Execute 2x the number of operations to ensure we use every device.
        for fut in [pg.allreduce(torch.ones(i + 1)).get_future() for i in range(4)]:
            fut.wait()

    def test_empty_tensors(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        xs = [torch.FloatTensor([])]
        fut = pg.broadcast(xs).get_future()
        fut.wait()
        output = fut.value()
        self.assertEqual(0, output[0].numel())
        self.assertEqualIgnoreType(xs[0], output[0])

    def test_broadcast_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = -1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.broadcast([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.broadcast([t1, t3], opts)

    def _test_broadcast_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            fut = pg.broadcast(xs, opts).get_future()
            fut.wait()
            return fut.value()

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = fn(torch.tensor([self.rank]))
            output = broadcast([x], i, 0)
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(torch.tensor([i]), output[0])

            # Run with 2 input tensors
            num = 2
            for j in range(num):
                xs = [
                    fn(torch.tensor([self.rank * num + 0.0])),
                    fn(torch.tensor([self.rank * num + 1.0])),
                ]

                output = broadcast(xs, i, j)
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(torch.tensor([i * num + j]), output[0])
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(torch.tensor([i * num + j]), output[1])

        # Test overloaded convenience function
        x = torch.tensor([self.rank + 1.0])
        fut = pg.broadcast(x, root=0).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(torch.tensor([1.0]), result[0])

    def test_broadcast_basics(self):
        self._test_broadcast_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    def test_broadcast_basics_cuda(self):
        self._test_broadcast_basics(lambda t: t.clone().cuda())

    def _test_broadcast_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        work_handles = [
            pg.broadcast(inputs[i], root=(i % self.world_size))
            for i in range(len(inputs))
        ]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.tensor([(i * self.world_size) + (i % self.world_size)]),
                inputs[i],
                msg=("Mismatch in iteration %d" % i),
            )

    def test_broadcast_stress(self):
        inputs = [torch.tensor([i * self.world_size + self.rank]) for i in range(1000)]
        self._test_broadcast_stress(inputs)

    @skip_if_lt_x_gpu(2)
    def test_broadcast_stress_cuda(self):
        inputs = [
            torch.tensor([i * self.world_size + self.rank]).cuda() for i in range(1000)
        ]
        self._test_broadcast_stress(inputs)

    def test_allreduce_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t3], opts)

    def _test_allreduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for (op, input, expected) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            fut = pg.allreduce([tensor], opts).get_future()
            fut.wait()
            result = fut.value()
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(expected, result[0])

        # Multi input tests
        tests = simple_multi_input_reduce_tests(self.rank, self.world_size)
        for (op, inputs, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensors = [fn(input) for input in inputs]
            fut = pg.allreduce(tensors, opts).get_future()
            fut.wait()
            result = fut.value()
            for tensor in result:
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(output, tensor)

        # Test overloaded convenience function (defaults to using sum)
        x = fn(torch.tensor([self.rank + 1.0]))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]), result[0]
        )

    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    def test_allreduce_basics_cuda(self):
        self._test_allreduce_basics(lambda t: t.clone().cuda())

    # _using_work_api tests are to make sure we still properly support work API.
    # This should go away as we deprecate it.
    def _test_allreduce_basics_using_work_api(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for (op, input, expected) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            work = pg.allreduce([tensor], opts)
            work.wait()
            result = work.result()
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(expected, result[0])

        # Multi input tests
        tests = simple_multi_input_reduce_tests(self.rank, self.world_size)
        for (op, inputs, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensors = [fn(input) for input in inputs]
            work = pg.allreduce(tensors, opts)
            work.wait()
            result = work.result()
            for tensor in result:
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(output, tensor)

        # Test overloaded convenience function (defaults to using sum)
        x = fn(torch.tensor([self.rank + 1.0]))
        work = pg.allreduce(x)
        work.wait()
        result = work.result()
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]), result[0]
        )

    def test_allreduce_basics_using_work_api(self):
        self._test_allreduce_basics_using_work_api(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    def test_allreduce_basics_cuda_using_work_api(self):
        self._test_allreduce_basics_using_work_api(lambda t: t.clone().cuda())

    def _test_allreduce_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = [pg.allreduce(inputs[i]).get_future() for i in range(len(inputs))]
        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(
                torch.tensor(
                    [
                        (i * self.world_size)
                        + (self.world_size * (self.world_size - 1) / 2)
                    ]
                ),
                future_handle.value()[0],
                msg=("Mismatch in iteration %d" % i),
            )

    def test_allreduce_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_allreduce_stress(inputs)

    @skip_if_lt_x_gpu(2)
    def test_allreduce_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allreduce_stress(inputs)

    def test_allreduce_coalesced_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros(1, dtype=torch.float32)
        t2 = torch.zeros(1, dtype=torch.float64)
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(1,))

        with self.assertRaisesRegex(ValueError, "requires non-empty tensor list"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([], opts)

        with self.assertRaisesRegex(ValueError, "tensors must all have the same type"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor layout at index"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1, t3], opts)

        with self.assertRaisesRegex(ValueError, "unsupported layout"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t3, t3.clone()], opts)

    @skip_if_lt_x_gpu(1)
    def test_allreduce_coalesced_checks_cuda(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros(1, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "unsupported device type"):
            opts = c10d.AllreduceCoalescedOptions()
            pg.allreduce_coalesced([t1.cuda(), t1.cuda()], opts)

    def _test_allreduce_coalesced_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        test_cases = simple_coalesced_reduce_tests(self.rank, self.world_size)
        for op, inputs, outputs in test_cases:
            opts = c10d.AllreduceCoalescedOptions()
            opts.reduceOp = op
            tensors = [fn(x) for x in inputs]
            fut = pg.allreduce_coalesced(tensors, opts).get_future()
            fut.wait()
            result = fut.value()
            for result_tensor, expected in zip(result, outputs):
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(result_tensor, expected)

    def test_allreduce_coalesced_basics(self):
        self._test_allreduce_coalesced_basics(lambda t: t.clone())

    def _test_allreduce_coalesced_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = [pg.allreduce_coalesced(input).get_future() for input in inputs]
        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(
                2
                * [
                    torch.tensor(
                        [
                            (i * self.world_size)
                            + (self.world_size * (self.world_size - 1) / 2)
                        ]
                    )
                ],
                result,
                msg="Mismatch in interation {}".format(i),
            )

    def test_allreduce_coalesced_stress(self):
        inputs = [2 * [torch.tensor([i + self.rank])] for i in range(1000)]
        self._test_allreduce_coalesced_stress(inputs)

    def test_sparse_allreduce_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1])
        t2 = torch.sparse_coo_tensor([[0]], [1], size=(2,))
        t3 = torch.sparse_coo_tensor([[0]], [1], size=(4,))

        with self.assertRaisesRegex(ValueError, "requires non-empty tensor list"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor layout"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t1, t2], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.AllreduceOptions()
            pg.allreduce([t2, t3], opts)

        # Sparse allreduce only works with c10d.ReduceOp.SUM.
        for op in [c10d.ReduceOp.PRODUCT, c10d.ReduceOp.MIN, c10d.ReduceOp.MAX]:
            with self.assertRaisesRegex(ValueError, "unsupported reduction operation"):
                opts = c10d.AllreduceOptions()
                opts.reduceOp = op
                pg.allreduce([t3], opts)

    def _test_sparse_allreduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        for num_inputs_per_rank in [1, 2]:
            tests = simple_sparse_reduce_tests(
                self.rank, self.world_size, num_inputs=num_inputs_per_rank
            )
            for (inputs, outputs) in tests:
                tensors = [fn(input) for input in inputs]
                fut = pg.allreduce(tensors).get_future()
                fut.wait()
                result = fut.value()
                self.assertEqual(tensors, outputs)
                self.assertEqual(result, outputs)

    def test_sparse_allreduce_basics(self):
        self._test_sparse_allreduce_basics(lambda t: t)

    @skip_if_lt_x_gpu(2)
    def test_sparse_allreduce_basics_cuda(self):
        self._test_sparse_allreduce_basics(lambda t: t.clone().cuda())

    def test_scatter_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = -1
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.world_size
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element output tensor list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([], [], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element output tensor list"
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = 0
            pg.scatter([t1, t1], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [], opts)

        with self.assertRaisesRegex(ValueError, "requires a single-element input list"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * self.world_size, [t1] * self.world_size], opts)

        desired_list_size = self.world_size
        incorrect_list_size = self.world_size - 1
        err_str = "Incorrect input list size {}. Input list size should be {}"
        with self.assertRaisesRegex(
            ValueError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * incorrect_list_size], opts)

        incorrect_list_size = self.world_size + 1
        with self.assertRaisesRegex(
            ValueError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t1] * incorrect_list_size], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t2] * self.world_size], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.ScatterOptions()
            opts.rootRank = self.rank
            pg.scatter([t1], [[t3] * self.world_size], opts)

        with self.assertRaisesRegex(ValueError, "requires empty input on non-root"):
            opts = c10d.ScatterOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.scatter([t1], [[t1] * self.world_size], opts)

    def _test_scatter_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        input = [fn(torch.tensor([self.rank])) for _ in range(self.world_size)]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the scatter root and accumulate work items
        futures = []
        for i in range(self.world_size):
            opts = c10d.ScatterOptions()
            opts.rootRank = i
            if i == self.rank:
                futures.append(pg.scatter([outputs[i]], [input], opts).get_future())
            else:
                futures.append(pg.scatter([outputs[i]], [], opts).get_future())

        # Wait for work to complete
        for i in range(self.world_size):
            futures[i].wait()
            result = futures[i].value()
            self.assertEqual(torch.tensor([i]), result[0])

    def test_scatter_basics(self):
        self._test_scatter_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    def test_scatter_basics_cuda(self):
        self._test_scatter_basics(lambda t: t.clone().cuda())

    def _test_scatter_stress(self, inputs, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        outputs = [
            [fn(torch.tensor([-1])) for _ in range(self.world_size)]
            for _ in range(len(inputs))
        ]
        future_handles = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ScatterOptions()
                opts.rootRank = root
                if root == self.rank:
                    fut = pg.scatter(
                        [outputs[i][root]], [[fn(e) for e in inputs[i]]], opts
                    ).get_future()
                else:
                    fut = pg.scatter([outputs[i][root]], [], opts).get_future()
                future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            result = future_handle.value()

            self.assertEqual(
                torch.tensor([iter + root]),
                result[0],
                msg=("Mismatch in iteration %d for rank %d" % (iter, root)),
            )

    def test_scatter_stress(self):
        inputs = [
            [torch.tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_scatter_stress(inputs, lambda t: t.clone())

    @unittest.skip("Test is flaky, see https://github.com/pytorch/pytorch/issues/15963")
    @skip_if_lt_x_gpu(2)
    def test_scatter_stress_cuda(self):
        inputs = [
            [torch.tensor([i + self.rank]) for _ in range(self.world_size)]
            for i in range(1000)
        ]
        self._test_scatter_stress(inputs, lambda t: t.clone().cuda())

    def test_gather_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = -1
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.world_size
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element input tensor list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element input tensor list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = 0
            pg.gather([], [t1, t1], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element output list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([], [t1], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element output list"
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * self.world_size, [t1] * self.world_size], [t1], opts)

        desired_list_size = self.world_size
        incorrect_list_size = self.world_size - 1
        err_str = "Incorrect output list size {}. Output list size should be {}"
        with self.assertRaisesRegex(
            ValueError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * incorrect_list_size], [t1], opts)

        incorrect_list_size = self.world_size + 1
        with self.assertRaisesRegex(
            ValueError, err_str.format(incorrect_list_size, desired_list_size)
        ):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t1] * incorrect_list_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t2] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            opts = c10d.GatherOptions()
            opts.rootRank = self.rank
            pg.gather([[t3] * self.world_size], [t1], opts)

        with self.assertRaisesRegex(ValueError, "requires empty output on non-root"):
            opts = c10d.GatherOptions()
            opts.rootRank = (self.rank + 1) % self.world_size
            pg.gather([[t1] * self.world_size], [t1], opts)

    def _test_gather_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        input = [fn(torch.tensor([self.rank]))]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the gather root and accumulate work items
        futures = []
        for i in range(self.world_size):
            opts = c10d.GatherOptions()
            opts.rootRank = i
            if i == self.rank:
                futures.append(pg.gather([outputs], input, opts).get_future())
            else:
                futures.append(pg.gather([], input, opts).get_future())

        # Wait for work to complete
        expected = [torch.tensor([rank]) for rank in range(self.world_size)]
        for i in range(self.world_size):
            futures[i].wait()
            result = futures[i].value()
            if i == self.rank:
                self.assertEqual(expected, result)

    def test_gather_basics(self):
        self._test_gather_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    def test_gather_basics_cuda(self):
        self._test_gather_basics(lambda t: t.clone().cuda())

    def _test_gather_stress(self, inputs, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = []
        outputs = [
            [[fn(torch.tensor([-1])) for _ in range(self.world_size)]]
            for _ in range(len(inputs))
        ]
        expected_outputs = [
            [[torch.tensor([i + j]) for j in range(self.world_size)]]
            for i in range(len(inputs))
        ]
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.GatherOptions()
                opts.rootRank = root
                if root == self.rank:
                    fut = pg.gather(outputs[i], [fn(inputs[i])], opts).get_future()
                else:
                    fut = pg.gather([], [fn(inputs[i])], opts).get_future()
                future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                result = future_handle.value()
                self.assertEqual(
                    expected_outputs[iter],
                    [result],
                    msg=("Mismatch in iteration %d for root %d" % (iter, root)),
                )

    def test_gather_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    def test_gather_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_gather_stress(inputs, lambda t: t.clone().cuda())

    def test_allgather_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "requires non-empty input tensor list"):
            pg.allgather([], [])

        with self.assertRaisesRegex(
            ValueError, "requires input/output tensor lists to have the same length"
        ):
            pg.allgather([], [t1])

        with self.assertRaisesRegex(
            ValueError, "requires input/output tensor lists to have the same length"
        ):
            pg.allgather([[t1] * self.world_size, [t1] * self.world_size], [t1])

        with self.assertRaisesRegex(ValueError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size - 1)], [t1])

        with self.assertRaisesRegex(ValueError, "invalid output tensor list"):
            pg.allgather([[t1] * (self.world_size + 1)], [t1])

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            pg.allgather(
                [[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t2]
            )

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            pg.allgather(
                [[t1, t1] * (self.world_size), [t1, t1] * (self.world_size)], [t1, t3]
            )

        with self.assertRaisesRegex(ValueError, "invalid tensor type"):
            pg.allgather([([t1, t2] * (self.world_size))[: self.world_size]], [t1])

        with self.assertRaisesRegex(ValueError, "invalid tensor size"):
            pg.allgather([([t1, t3] * (self.world_size))[: self.world_size]], [t1])

    def _test_allgather_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Run with N input tensor per rank
        for n in [1, 2, 3]:
            input = [fn(torch.tensor([n * self.rank + i])) for i in range(n)]
            output = [
                [fn(torch.tensor([-1])) for _ in range(n * self.world_size)]
                for _ in range(n)
            ]
            expected_output = [
                [torch.tensor([i]) for i in range(n * self.world_size)]
                for _ in range(n)
            ]
            fut = pg.allgather(output, input).get_future()
            fut.wait()
            result = fut.value()
            if n == 1:
                result = [result]
            self.assertEqual(expected_output, result)

    def test_allgather_basics(self):
        self._test_allgather_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    def test_allgather_basics_cuda(self):
        self._test_allgather_basics(lambda t: t.clone().cuda())

    def _test_allgather_stress(self, inputs, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = []
        outputs = [
            [[fn(torch.tensor([-1])) for _ in range(self.world_size)]]
            for _ in range(len(inputs))
        ]
        expected_outputs = [
            [[torch.tensor([i + j]) for j in range(self.world_size)]]
            for i in range(len(inputs))
        ]
        for i in range(len(inputs)):
            fut = pg.allgather(outputs[i], [fn(inputs[i])]).get_future()
            future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            self.assertEqual(
                expected_outputs[i],
                [result],
                msg=("Mismatch in iteration %d" % i),
            )

    def test_allgather_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    def test_allgather_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_allgather_stress(inputs, lambda t: t.clone().cuda())

    def test_allgather_coalesced_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())
        dummy_input = [torch.zeros([1], dtype=torch.float32)]
        dummy_output_lists = [
            [torch.zeros([1], dtype=torch.float32)] for _ in range(self.world_size)
        ]

        # One of output tensors does not match input list.
        dummy_output_lists[0] = [torch.zeros([0], dtype=torch.float32)]
        with self.assertRaisesRegex(
            ValueError, "invalid size of output tensor at index 0"
        ):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # One of output tensors does not match input list.
        dummy_output_lists[0] = [torch.zeros([1], dtype=torch.float64)]
        with self.assertRaisesRegex(ValueError, "invalid tensor type at index 0"):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # Output lists have too many elements
        dummy_output_lists = [
            [torch.zeros([1], dtype=torch.float32)] for _ in range(self.world_size + 1)
        ]
        with self.assertRaisesRegex(
            ValueError, "output lists should be equal to world size"
        ):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

        # Output is not a list of lists.
        dummy_output_lists = [torch.zeros([0], dtype=torch.float32)]
        with self.assertRaisesRegex(
            RuntimeError, "Invalid function argument.*output_tensor_lists"
        ):
            c10d.all_gather_coalesced(dummy_output_lists, dummy_input, pg)

    def test_reduce_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        t1 = torch.zeros([1], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root rank"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.world_size
            opts.rootTensor = 0
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(ValueError, "invalid root tensor"):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 1
            pg.reduce([t1], opts)

        with self.assertRaisesRegex(
            ValueError, "requires a single-element tensor list"
        ):
            opts = c10d.ReduceOptions()
            opts.rootRank = self.rank
            opts.rootTensor = 0
            pg.reduce([t1, t1], opts)

    def _test_reduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())
        for (op, input, output) in simple_reduce_tests(self.rank, self.world_size):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                tmp = fn(input)
                fut = pg.reduce([tmp], opts).get_future()
                fut.wait()
                result = fut.value()
                if root == self.rank:
                    # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                    self.assertEqualIgnoreType(output, result[0])

    def test_reduce_basics(self):
        self._test_reduce_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    def test_reduce_basics_cuda(self):
        self._test_reduce_basics(lambda t: t.clone().cuda())

    def _test_reduce_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, self.opts(threads=8)
        )
        future_handles = []
        outputs = []
        for i in range(len(inputs)):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.rootRank = root
                tmp = inputs[i].clone()
                outputs.append(tmp)
                fut = pg.reduce([tmp], opts).get_future()
                future_handles.append(fut)

        for i, future_handle in enumerate(future_handles):
            future_handle.wait()
            result = future_handle.value()
            iter = i // self.world_size
            root = i % self.world_size
            if root == self.rank:
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(
                    torch.tensor(
                        [
                            (iter * self.world_size)
                            + (self.world_size * (self.world_size - 1) / 2)
                        ]
                    ),
                    result[0],
                    msg=("Mismatch in iteration %d with root rank %d" % (iter, root)),
                )

    def test_reduce_stress(self):
        inputs = [torch.tensor([i + self.rank]) for i in range(1000)]
        self._test_reduce_stress(inputs)

    @skip_if_lt_x_gpu(2)
    def test_reduce_stress_cuda(self):
        inputs = [torch.tensor([i + self.rank]).cuda() for i in range(1000)]
        self._test_reduce_stress(inputs)

    def test_send_recv_all_to_all(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Preallocate tensors for input/output
        inputs = [torch.tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.tensor([-1]) for _ in range(self.world_size)]

        # Issue sends
        send_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            send_work.append(pg.send([inputs[i]], i, 0))

        # Issue recvs
        recv_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            recv_work.append(pg.recv([outputs[i]], i, 0))

        # Wait for sends to complete
        for work in send_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Wait for recvs to complete
        for work in recv_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Test that every output other than our own contains the respective rank
        for i in range(self.world_size):
            if i == self.rank:
                continue
            self.assertEqual(torch.tensor([i]), outputs[i])

    def test_barrier_implies_wait(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupGloo(store, self.rank, self.world_size, self.opts())

        # Kick off allreduce operations
        size = (100, 100)
        num = 16
        tensors = [torch.full(size, float(i)) for i in range(num)]
        for tensor in tensors:
            # Note: leak the returned work handle
            pg.allreduce(tensor)

        # Barrier should ensure all previous work has completed
        pg.barrier().get_future().wait()

        for i, tensor in enumerate(tensors):
            self.assertEqual(torch.full(size, float(i * self.world_size)), tensor)

    @skip_if_win32()
    def test_round_robin(self):
        num_process_groups = 2
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d._round_robin_process_groups(
            [
                c10d.ProcessGroupGloo(
                    c10d.PrefixStore(str(i), store), self.rank, self.world_size, self.opts()
                )
                for i in range(num_process_groups)
            ]
        )

        # Run a few collectives so that we have called each process group
        for _ in range(num_process_groups + 1):
            tensor = torch.full([100, 100], float(self.rank))
            pg.broadcast(tensor, root=0).wait()
            self.assertEqual(torch.full([100, 100], 0.0), tensor)

    @skip_if_win32()
    def test_round_robin_create_destroy(self):
        store = c10d.FileStore(self.file_name, self.world_size)

        def create(num, prefix):
            return c10d._round_robin_process_groups(
                [
                    c10d.ProcessGroupGloo(
                        c10d.PrefixStore("%s/%d" % (prefix, i), store),
                        self.rank,
                        self.world_size,
                        self.opts()
                    )
                    for i in range(num)
                ]
            )

        # Run create/use/destroy twice
        for i in range(2):
            num_process_groups = 2
            pg = create(num=num_process_groups, prefix=i)
            for _ in range(3):
                tensor = torch.ones([10, 10])
                pg.allreduce(tensor).wait()
                self.assertEqual(torch.full([10, 10], float(self.world_size)), tensor)
            del pg


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class DistributedDataParallelTest(test_c10d_common.AbstractDistributedDataParallelTest, MultiProcessTestCase):

    def setUp(self):
        super(DistributedDataParallelTest, self).setUp()
        if sys.platform == "win32":
            self._spawn_processes()
        else:
            self._fork_processes()

    def _test_gloo_backend(
            self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        store = c10d.FileStore(self.file_name, self.world_size)
        options = c10d.ProcessGroupGloo._Options()
        options._devices = [create_device(interface=LOOPBACK)]
        process_group = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, options
        )
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_gloo()
    def test_gloo_backend_cpu_module(self):
        self._test_gloo_backend([torch.device("cpu")], None)

    @requires_gloo()
    def test_gloo_backend_cpu_module_grad_is_view(self):
        self._test_gloo_backend([torch.device("cpu")], None, gradient_as_bucket_view=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_gloo_backend_1gpu_module_device_ids_integer_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, int_devices)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_gloo_backend_1gpu_module_device_ids_torch_device_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, devices)

    @requires_gloo()
    @skip_if_lt_x_gpu(4)
    def test_gloo_backend_2gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, None, multi_device=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(8)
    def test_gloo_backend_4gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_gloo_backend(devices, None, multi_device=True)

    def _test_global_local_unused_params_grad(self, gradient_as_bucket_view=False, static_graph=False):
        """
        By simulating a multi-task training, this test is to make sure:
        1) DDP does not touch the grad of globally unused parameters.
        2) DDP does update the grad of locally unused parameters.
        """

        class GlobalLocalUnusedParamModule(nn.Module):
            def __init__(self):
                super(GlobalLocalUnusedParamModule, self).__init__()
                self.t0 = Task()
                self.t1 = Task()
                self.task_unused = Task()

            def task_parameters(self):
                return (self.t0.p, self.t1.p, self.task_unused.p)

            def forward(self, x, rank):
                return self.t0(x) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            # Run forward
            output = model(8, self.rank)

            # The grads of all parameters should be None at this point.
            t0_p, t1_p, task_unused_p = model.module.task_parameters()
            self.assertIsNone(t0_p.grad)
            self.assertIsNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

            # Run backward
            output.mean().backward()

            # Now locally unused parameter should have grad updated on all ranks.
            # However the globally unused parameter should still have None grad.
            self.assertIsNotNone(t0_p.grad)
            self.assertIsNotNone(t1_p.grad)
            self.assertIsNone(task_unused_p.grad)

        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Test on CPU
        cpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
        if static_graph:
            cpu_model._set_static_graph()
        run_and_verify_grad(cpu_model)

        # Test on GPU
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            GlobalLocalUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
        if static_graph:
            gpu_model._set_static_graph()
        run_and_verify_grad(gpu_model)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad(self):
        self._test_global_local_unused_params_grad()

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_grad_is_view(self):
        self._test_global_local_unused_params_grad(gradient_as_bucket_view=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_global_local_unused_params_grad_with_static_graph(self):
        self._test_global_local_unused_params_grad(static_graph=True)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_find_unused_parameters_when_unused_parameters_empty(self):
        """
        An empty unused_parameters array does not imply find_unused_parameters =
        false. This test makes sure that DDP allreduces unused parameters
        accordingly where the forward pass in some process uses all parameters.
        This unit test creates a module that uses all parameters in rank = 0, and
        has unused parameters in other ranks.
        """

        class FindUnusedParamModule(nn.Module):
            def __init__(self):
                super(FindUnusedParamModule, self).__init__()
                self.t0 = Task()
                self.t1 = Task()

            def task_parameters(self):
                return (self.t0.p, self.t1.p)

            def forward(self, x, rank):
                return self.t1(self.t0(x)) if rank == 0 else self.t1(x)

        def run_and_verify_grad(model):
            # Run forward
            output = model(8, self.rank)

            # The grads of all parameters should be None at this point.
            [self.assertIsNone(t_p.grad) for t_p in model.module.task_parameters()]

            # Run backward
            output.mean().backward()

            # Now locally unused parameter should have grad updated on all ranks.
            [self.assertIsNotNone(t_p.grad) for t_p in model.module.task_parameters()]

        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Test on CPU
        cpu_model = DistributedDataParallel(
            FindUnusedParamModule().cpu(),
            process_group=process_group,
            find_unused_parameters=True,
        )
        run_and_verify_grad(cpu_model)

        # Test on GPU
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            FindUnusedParamModule().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            find_unused_parameters=True,
        )
        run_and_verify_grad(gpu_model)

    @requires_gloo()
    def test_ignored_output(self):
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        class IgnoredOutput(nn.Module):
            def __init__(self):
                super(IgnoredOutput, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        model = DistributedDataParallel(
            IgnoredOutput().float(),
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # Run a few iterations where we ignore the output.
        for _ in range(4):
            output = model(input)
            del output

        # Run a few iterations where we use the output.
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_gloo()
    def test_ignored_output_with_unused_parameters(self):
        """
        Test that the output of a model can be ignored and that there is no
        implicit requirement that `backward` gets called, if not all model
        parameters participated in computing the model output.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        class IgnoredOutputWithUnusedParameters(nn.Module):
            def __init__(self):
                super(IgnoredOutputWithUnusedParameters, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        model = DistributedDataParallel(
            IgnoredOutputWithUnusedParameters().float(),
            process_group=process_group,
            find_unused_parameters=True,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

        # Run a few iterations where we ignore the output.
        for _ in range(4):
            output = model(input)
            del output

        # Run a few iterations where we use the output.
        for _ in range(4):
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

    def _run_and_verify_sparse_gradients(self, vanilla_model, ddp_model):
        mult = 2
        batch_size = mult * self.world_size
        criterion = nn.CrossEntropyLoss()
        input = torch.randint(0, 10, [batch_size, 2])
        target = torch.randint(0, 10, [batch_size])

        # Run with entire batch against single process version
        criterion(vanilla_model(input), target).backward()

        # Run with partial batch against multi process version
        partial_input = input.split(mult)[self.rank]
        partial_target = target.split(mult)[self.rank]
        criterion(ddp_model(partial_input), partial_target).backward()

        # Check that the gradients are sparse and identical
        vanilla_parameter = next(vanilla_model.parameters())
        ddp_parameter = next(ddp_model.parameters())
        self.assertEqual(vanilla_parameter.grad, ddp_parameter.grad)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_save_load_checkpoint(self):
        dist.init_process_group(
            "gloo",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

        class TestModel(nn.Module):
            def __init__(self):
                super(TestModel, self).__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        def train_loop(model, optimizer, iterations):
            for _ in range(iterations):
                optimizer.zero_grad()
                output = model(input)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        device_id = gpus_for_rank(self.world_size)[self.rank][0]

        model_withload = TestModel().float().to(device_id)
        model_withoutload = TestModel().float().to(device_id)

        ddp_withload = DistributedDataParallel(
            model_withload,
            device_ids=[device_id],
        )
        ddp_withoutload = DistributedDataParallel(
            model_withoutload,
            device_ids=[device_id],
        )

        # ensure that all the three models start with the same set of parameters. By default they are randomized on construction
        for p in ddp_withload.parameters():
            with torch.no_grad():
                p.zero_()
        for p in model_withload.parameters():
            with torch.no_grad():
                p.zero_()
        for p in ddp_withoutload.parameters():
            with torch.no_grad():
                p.zero_()

        batch_size = 4
        criterion = nn.CrossEntropyLoss()

        optimizer_withload = torch.optim.SGD(ddp_withload.parameters(), lr=0.001)
        optimizer_non_ddp_withload = torch.optim.SGD(model_withload.parameters(), lr=0.001)
        optimizer_withoutload = torch.optim.SGD(ddp_withoutload.parameters(), lr=0.001)

        input = torch.rand([batch_size, 2], dtype=torch.float).to(device_id)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # run the model for 6 iterations, with a checkpoint in the middle
        train_loop(ddp_withload, optimizer_withload, 3)

        # zero out parameters of both DDP and non-DDP models and reload them from the DDP state dict
        checkpoint_path = tempfile.gettempdir() + "/model.checkpoint"
        if self.rank == 0:
            torch.save(ddp_withload.state_dict(), checkpoint_path)

        dist.barrier()
        map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
        ddp_state_dict = torch.load(checkpoint_path, map_location=map_location)

        for model in [ddp_withload, model_withload]:
            for p in ddp_withload.parameters():
                with torch.no_grad():
                    p.zero_()
        ddp_withload.load_state_dict(ddp_state_dict)
        # the non-DDP model needs to first remove the prefix of "module." from the DDP state dict
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(ddp_state_dict, "module.")
        model_withload.load_state_dict(ddp_state_dict)

        train_loop(ddp_withload, optimizer_withload, 3)
        train_loop(model_withload, optimizer_non_ddp_withload, 3)

        # re-run the model with the same inputs for 6 iterations with no checkpoint
        train_loop(ddp_withoutload, optimizer_withoutload, 6)

        for p_withload, p_withoutload, p_non_ddp_withload in zip(
            ddp_withload.parameters(), ddp_withoutload.parameters(), model_withload.parameters()
        ):
            self.assertEqual(p_withload, p_withoutload)
            self.assertEqual(p_non_ddp_withload, p_withoutload)

    def _test_sparse_gradients(self, gradient_as_bucket_view=False):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Ensure initialized weights and inputs are identical across processes
        torch.manual_seed(1337)

        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)

    @requires_gloo()
    def test_sparse_gradients(self):
        self._test_sparse_gradients()

    @requires_gloo()
    def test_sparse_gradients_grad_is_view(self):
        self._test_sparse_gradients(gradient_as_bucket_view=True)

    @requires_gloo()
    def test_ddp_comm_hook_future_passing_cpu(self):
        """
        This unit test verifies whether the Future object is passed properly.
        The callback function creates a Future object and sets a value to it.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Test on CPU
        cpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().cpu(), process_group=process_group
        )

        # Register DDP Communication Hook
        cpu_model.register_comm_hook(None, self._simple_hook)

        # check whether the grads are equal to what then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(cpu_model, 8, 2 * torch.ones(2, 2))

    def _gpu_model_with_ddp_comm_hook(
            self, process_group, hook=None, gradient_as_bucket_view=False, state=None
    ):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Register a DDP communication hook if any.
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_gloo(self):
        """
        This unit test verifies whether the Future object is passed properly using gloo backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Get GPU model with simple_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # check whether the grads are equal to what simple_hook's then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    @requires_gloo()
    def test_ddp_invalid_comm_hook_init(self):
        """
        This unit test makes sure that register_comm_hook properly checks the format
        of hook defined by user. The Python hook must be callable. This test also
        checks whether bucket annotation checked properly if defined.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        with self.assertRaisesRegex(TypeError, "Communication hook must be callable."):
            model.register_comm_hook(state=None, hook=1)

        with self.assertRaisesRegex(
                ValueError, "bucket annotation should be dist.GradBucket."
        ):
            def comm_hook(state: object, bucket: int) -> torch.futures.Future:
                return torch.futures.Future()

            model.register_comm_hook(state=None, hook=comm_hook)

    @requires_gloo()
    def test_ddp_invalid_comm_hook_return_type(self):
        """
        This test checks whether return annotation checked properly if defined. It also
        checks whether an internal error is thrown if return type is incorrect and user
        hasn't specified any return type annotation.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        with self.assertRaisesRegex(
                ValueError,
                "Communication hook: return annotation should be torch.futures.Future or torch._C.Future.",
        ):
            def comm_hook(state: object, bucket: dist.GradBucket) -> int:
                return torch.futures.Future()

            model.register_comm_hook(state=None, hook=comm_hook)

        with self.assertRaisesRegex(
                RuntimeError,
                "callback must return a torch.futures.Future or torch._C.Future object, but got",
        ):
            def comm_hook(state: object, bucket: dist.GradBucket):
                return 1

            model.register_comm_hook(state=None, hook=comm_hook)

            # Run forward
            output = model(8, self.rank)

            # Run backward
            output.mean().backward()

    @requires_gloo()
    def test_ddp_comm_hook_register_just_once(self):
        """
        DDP communication hook can only be registered once. This test validates whether
        the error is thrown properly when register_comm_hook is called more than once.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        model = DistributedDataParallel(
            ModuleForDdpCommHook(), process_group=process_group
        )

        def dummy_hook(state, bucket):
            fut = torch.futures.Future()
            fut.set_result([bucket.get_tensor()])
            return fut

        model.register_comm_hook(None, dummy_hook)

        with self.assertRaisesRegex(
                RuntimeError,
                "register_comm_hook or register_builtin_comm_hook can only be called once.",
        ):
            model.register_comm_hook(None, dummy_hook)

    @requires_gloo()
    def test_ddp_comm_hook_sparse_gradients(self):
        """
        Runs "test_sparse_gradients" unit test with DDP communication hook. We define a
        simple hook that does allreduce and works with gloo backend for this test.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupGloo(store, self.rank, self.world_size)

        # Ensure initialized weights and inputs are identical across processes
        torch.manual_seed(1337)

        vanilla_model = SparseGradientModule()
        ddp_model = DistributedDataParallel(
            copy.deepcopy(vanilla_model),
            process_group=process_group,
        )

        def allreduce_hook_gloo(
                state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future:
            def div_by_world_size(fut):
                # Divide the result by 2 * world_size.
                return [t / self.world_size for t in fut.wait()]

            # Prepare allreduced grad bucket tensors by running an async work.
            fut = process_group.allreduce([bucket.get_tensor()]).get_future()
            return fut.then(div_by_world_size)

        ddp_model.register_comm_hook(None, allreduce_hook_gloo)

        self._run_and_verify_sparse_gradients(vanilla_model, ddp_model)


class ReducerModule(nn.Module):
    def __init__(self):
        super(ReducerModule, self).__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 4, bias=False)
        self.fc3 = nn.Linear(4, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, use_fc3=True):
        x = self.relu(self.fc1(x)).float()
        x = self.relu(self.fc2(x)).float()
        if use_fc3:
            x = self.fc3(x).float()
        return F.softmax(x, dim=1)


@requires_gloo()
class ReducerTest(TestCase):
    def setUp(self):
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.store = c10d.FileStore(self.file.name, 1)
        self.process_group = c10d.ProcessGroupGloo(self.store, 0, 1)

    def test_single_dtype_single_bucket(self):
        model = ReducerModule()
        parameters = list(model.parameters())
        buckets = [list(range(len(parameters)))]
        dist.Reducer([parameters], buckets, self.process_group)

    def _create_mixed_precision_model(self):
        model = ReducerModule()
        model.float()
        model.fc1.double()
        return model

    def test_multi_dtype_single_bucket(self):
        model = self._create_mixed_precision_model()

        # Raise if there are multiple types per bucket.
        # In this case we create one bucket for all parameters.
        with self.assertRaises(RuntimeError):
            parameters = [list(model.parameters())]
            buckets = [list(range(len(parameters[0])))]
            dist.Reducer(parameters, buckets, self.process_group)

    def test_multi_dtype_multi_bucket(self):
        model = self._create_mixed_precision_model()
        parameters = [list(model.parameters())]
        group_by_dtype = groupby(
            range(len(parameters[0])), key=lambda i: parameters[0][i].dtype
        )
        buckets = [list(indices) for _, indices in group_by_dtype]
        dist.Reducer(parameters, buckets, self.process_group)

    def _create_reducer_for_models(self, models, find_unused_parameters=False):
        parameters = [list(model.parameters()) for model in models]
        group_by_dtype = groupby(
            range(len(parameters[0])), key=lambda i: parameters[0][i].dtype
        )
        buckets = [list(indices) for _, indices in group_by_dtype]
        return dist.Reducer(
            parameters,
            buckets,
            self.process_group,
            find_unused_parameters=find_unused_parameters,
        )

    def test_reducer_no_multi_replicas(self):
        num_replicas = 2
        models = [self._create_mixed_precision_model() for _ in range(num_replicas)]
        with self.assertRaisesRegex(
                RuntimeError,
                "Expected exactly one model replica.",
        ):
            reducer = self._create_reducer_for_models(models)

    def test_forward_backward(self):
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model])
        reducer.prepare_for_forward()
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss(model(input), target)
        reducer.prepare_for_backward(output)
        output.backward()

    def test_forward_backward_unused_parameters(self):
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model], find_unused_parameters=True)
        reducer.prepare_for_forward()
        loss = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.double)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])
        output = loss(model(input, use_fc3=False), target)

        # Check that the grad of fc3 is not set.
        self.assertEqual(None, model.fc3.weight.grad)

        # Compute and accumulate gradients.
        reducer.prepare_for_backward(output)
        output.backward()

        # The reducer will have marked the grad of fc3 as ready, because
        # it doesn't show up in the autograd graph of `output`. Since fc3.weight
        # is considered being globally unused, it will be kept untouched as None.
        self.assertEqual(None, model.fc3.weight.grad)

    def test_forward_backward_optimizer(self):
        batch_size = 10
        model = self._create_mixed_precision_model()
        reducer = self._create_reducer_for_models([model], find_unused_parameters=True)
        reducer.prepare_for_forward()
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        for i in range(3):
            input = torch.rand([batch_size, 2], dtype=torch.double)
            target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)])

            # The `zero_grad` function calls `detach_` and `zero_` on the grad
            # tensors of model parameters. If we tried to set the grad tensors
            # to a view of the reducer's bucket tensors, this would blow up.
            optimizer.zero_grad()

            # Unused parameter only in the first iteration.
            output = loss(model(input, use_fc3=(i > 0)), target)
            reducer.prepare_for_backward(output)
            output.backward()
            optimizer.step()


@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):

    def setUp(self):
        super(CommTest, self).setUp()
        if sys.platform == "win32":
            self._spawn_processes()
        else:
            self._fork_processes()

    def tearDown(self):
        super(CommTest, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _test_broadcast_coalesced(self, process_group, device, root_rank):
        half = torch.float16

        # No support for float16 for CPU tensors
        if device == torch.device("cpu"):
            half = torch.float32

        target = torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

        # The tensors to pass to broadcast are idential to the target
        # only on the process that is the root of the broadcast.
        if self.rank == root_rank:
            tensors = list(tensor.clone() for tensor in target)
        else:
            tensors = list(torch.zeros_like(tensor) for tensor in target)

        if self.rank != root_rank:
            self.assertNotEqual(tensors, target)

        c10d._broadcast_coalesced(
            process_group, tensors, buffer_size=256, src=root_rank
        )

        if self.rank != root_rank:
            self.assertEqual(tensors, target)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_gloo_cuda(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        options = c10d.ProcessGroupGloo._Options()
        options._devices = [create_device(interface=LOOPBACK)]
        process_group = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, options
        )
        device = torch.device("cuda:%d" % self.rank)
        ranks = list(range(self.world_size))
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_gloo()
    def test_broadcast_coalesced_gloo_cpu(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        options = c10d.ProcessGroupGloo._Options()
        options._devices = [create_device(interface=LOOPBACK)]
        process_group = c10d.ProcessGroupGloo(
            store, self.rank, self.world_size, options
        )
        device = torch.device("cpu")
        ranks = list(range(self.world_size))
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_default_pg_gloo(self):
        self._test_sequence_num_set_default_pg(backend="gloo")

    @requires_gloo()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_gloo_new_group(self):
        self._test_sequence_num_set_new_group(backend="gloo")

    @skip_if_lt_x_gpu(2)
    @requires_gloo()
    def test_sequence_num_incremented_gloo_default(self):
        self._test_sequence_num_incremented_default_group("gloo")

    @skip_if_lt_x_gpu(4)
    @requires_gloo()
    def test_sequence_num_incremented_gloo_subgroup(self):
        if self.world_size < 4:
            return unittest.skip("Test requires world_size of at least 4")
        self._test_sequence_num_incremented_subgroup("gloo")

    @requires_gloo()
    def test_gloo_barrier_device_ids(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )

        with self.assertRaisesRegex(RuntimeError, "device_ids not supported"):
            c10d.barrier(device_ids=[self.rank])


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
