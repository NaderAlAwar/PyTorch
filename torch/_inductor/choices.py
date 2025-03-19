from __future__ import annotations

import functools
import logging
import typing
from collections import defaultdict
from typing import Any, Callable, TYPE_CHECKING

import sympy

import torch
from torch.utils._ordered_set import OrderedSet

from . import config, ir
from .codecache import write_text
from .ir import is_cpu, Pointwise, Reduction, StorageBox
from .memory import (
    reorder_for_peak_memory,
    topological_sort_bfs,
    topological_sort_dfs,
    topological_sort_lpmf,
)
from .metrics import get_metric_table, is_metric_table_enabled
from .runtime.hints import DeviceProperties, ReductionHint
from .scheduler import BaseSchedulerNode, Scheduler, SchedulerBuffer, WhyNoFuse
from .utils import cmp, has_free_symbols
from .virtualized import V


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .codegen.simd_kernel_features import SIMDKernelFeatures
    from .codegen.triton import TritonKernel
    from .graph import GraphModule


class Sortable(typing.Protocol):
    """Anything that can be used as a list.sort() key (int/tuple/etc)"""

    def __lt__(self, other: typing.Self) -> bool: ...


class InductorChoices:
    """
    This class contains a collection of default heuristics that effect performance of our generated
    code.  We try to not put correctness requirements in this file.

    You can override the choices made here by doing:

            class MyHeuristics(InductorChoices):
                ...

            torch._inductor.virtualized.V.set_choices_handler(MyHeuristics())
    """

    def triton_kernel_kwargs(
        self,
        kernel_cls: type[TritonKernel],
        features: SIMDKernelFeatures,
        groups: list[sympy.Expr],
        kernel_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook to change the kwargs passed to TritonKernel, used to apply fixed configurations"""
        return kernel_kwargs

    @staticmethod
    def should_use_cooperative_reduction(features: SIMDKernelFeatures) -> bool:
        """Heuristic to decide if a cooperative reduction should be used."""
        if config.triton.force_cooperative_reductions:
            return True
        if (
            not config.triton.cooperative_reductions
            or V.graph.get_current_device_or_throw().type == "cpu"
        ):
            return False

        xhint = V.graph.sizevars.size_hint(features.numel, fallback=2)
        if xhint <= 8:
            threshold = 32768 * xhint
        elif xhint <= 16:
            threshold = 2097152
        else:
            return False
        # TODO(jansel): should this default on for dynamic shapes?
        return V.graph.sizevars.statically_known_geq(
            features.reduction_numel, threshold
        )

    @staticmethod
    def should_use_persistent_reduction(
        features: SIMDKernelFeatures, cooperative_reduction: bool
    ) -> bool:
        """
        Heuristic to decide if a persistent reduction should be used.
        """
        if not config.triton.persistent_reductions:
            return False
        threshold = {
            ReductionHint.INNER: 1024,
        }.get(features.get_reduction_hint(), 64)

        if cooperative_reduction:
            # The RSPLIT of cooperative reductions means each thread block is operating on fewer elements
            try:
                threshold *= 32 // min(V.graph.sizevars.size_hint(features.numel), 32)
            except ValueError:
                pass  # unbacked symint

        # If multi_kernel is enabled, we do more aggressive persistent reduction.
        # This may result in some persistent reductions slower than the
        # corresponding non-persistent reductions. MultiKernel will do benchmarking
        # to pick the faster one.
        if config.triton.multi_kernel:
            threshold *= 16
        return V.graph.sizevars.statically_known_leq(
            features.reduction_numel, threshold
        )  # type: ignore[arg-types]

    @staticmethod
    def want_no_x_dim(features: SIMDKernelFeatures) -> bool:
        """
        Heuristic to decide if we should drop the X dimension from a persistent reduction kernel.
        So the [XBLOCK, RBLOCK] block becomes a [RBLOCK] block and XBLOCK is forced to be always 1.
        Strangely this is faster than a [1, RBLOCK] block in some cases.
        """
        return (
            features.get_reduction_hint() == ReductionHint.INNER
            and V.graph.sizevars.statically_known_geq(features.reduction_numel, 256)
        )

    @staticmethod
    def reduction_split_factor(
        device: torch.device,
        reduction_numel_hint: int,
        numel_hint: int,
        inner_reduction: bool,
    ) -> int:
        """Heuristic to decide the RSPLIT used for split reductions.
        When a reduction has a small number of outputs there is not enough parallelism,
        so we will do the reduction in two phases."""
        props = DeviceProperties.create(device)
        num_sm = props.multi_processor_count
        min_elements_per_thread = 32
        max_elements_per_thread = 512
        threads_per_sm = 2048
        min_elements_per_device = min_elements_per_thread * num_sm * threads_per_sm
        max_elements_per_device = max_elements_per_thread * num_sm * threads_per_sm
        num_warps = 8
        num_threads = 32 * num_warps

        if inner_reduction:
            # do heuristics that's close to eager mode for split inner reduction
            # we leak reduction autotune configs here, and will need to refactor to avoid this later
            if numel_hint >= 2 * num_sm:  # don't split if there are enough outputs
                return 1
            if reduction_numel_hint <= 8192:
                return 1
            if reduction_numel_hint * numel_hint <= min_elements_per_device:
                split_size = min_elements_per_thread
            elif reduction_numel_hint * numel_hint < max_elements_per_device:
                target_blocks = num_sm * threads_per_sm // (2 * num_threads)
                blocks_per_output = (target_blocks + numel_hint - 1) // numel_hint
                tmp_split_size = (
                    reduction_numel_hint + num_threads * blocks_per_output - 1
                ) // (num_threads * blocks_per_output)
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
                if abs(closest - tmp_split_size) < 30:
                    # prefer even splits, but never smalle than min_elements_per_thread
                    split_size = max(closest, min_elements_per_thread)
                else:
                    split_size = tmp_split_size
            else:
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
                if abs(closest - max_elements_per_thread) < 50:
                    # prefer even splits
                    split_size = closest
                else:
                    split_size = max_elements_per_thread
            return (reduction_numel_hint + split_size * num_threads - 1) // (
                split_size * num_threads
            )
        else:
            # TODO the best heuristic currently has XBLOCK (corresponding to numel_hint) 128
            # extend to even smaller number of outputs
            rvals_per_thread = 4  # comes from heuristics, refactor to not leak here
            xvals_per_block = 128
            xblocks = (numel_hint + xvals_per_block - 1) // xvals_per_block
            if reduction_numel_hint * numel_hint < min_elements_per_device:
                split_size = min_elements_per_thread
            elif reduction_numel_hint * numel_hint < max_elements_per_device:
                target_blocks = num_sm * threads_per_sm // (num_threads)
                target_blocks = (target_blocks + xblocks - 1) // xblocks
                tmp_split_size = (
                    reduction_numel_hint + rvals_per_thread * target_blocks - 1
                ) // (rvals_per_thread * target_blocks)
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - tmp_split_size))
                if abs(tmp_split_size - closest) < 20:
                    split_size = max(closest, min_elements_per_thread)
                else:
                    split_size = tmp_split_size
            else:
                divisors = sympy.divisors(reduction_numel_hint)
                closest = min(divisors, key=lambda x: abs(x - max_elements_per_thread))
                if abs(closest - max_elements_per_thread) < 50:
                    # prefer even splits
                    split_size = closest
                else:
                    split_size = max_elements_per_thread

            return (reduction_numel_hint + rvals_per_thread * split_size - 1) // (
                rvals_per_thread * split_size
            )

    @staticmethod
    def can_fuse(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """
        Heuristics to prevent fusion applied to both horizontal and vertical fusions.  Heuristics here should not
        be needed for correctness and tweaking them may yield additional performance.

        See also some related heuristics that can be changed via config:
            - config.triton.tiling_prevents_pointwise_fusion
            - config.triton.tiling_prevents_reduction_fusion
            - config.aggressive_fusion (will cause this function to be called more times)
        """
        if shared_data_score == 0 and (
            not config.aggressive_fusion or node1.is_reduction() or node2.is_reduction()
        ):
            if is_metric_table_enabled("fusion_failure_due_to_indexing_mismatch"):
                common_buf_names: OrderedSet[str] = (
                    node1.read_writes.buffer_names() & node2.read_writes.buffer_names()
                )
                if len(common_buf_names) > 0:
                    get_metric_table("fusion_failure_due_to_indexing_mismatch").add_row(
                        lambda: {
                            "pre_grad_graph_id": V.graph.graph_id,
                            "post_grad_graph_id": V.graph.post_grad_graph_id,
                            "node1_name": node1.get_name(),
                            "node2_name": node2.get_name(),
                            "node1_debug_str": write_text(node1.debug_str()),
                            "node2_debug_str": write_text(node2.debug_str()),
                            "common_buffer_names": list(common_buf_names),  # type: ignore[dict-item]
                            "failure_reason": scheduler.decide_fusion_fail_reason(
                                node1, node2, common_buf_names
                            ),
                        }
                    )

                    WhyNoFuse(node1, node2)("no shared data due to indexing mismatch")
                    return False
            WhyNoFuse(node1, node2)("no shared data")
            return False  # heuristic not needed for correctness

        if (
            not node1.is_foreach()
            and not node2.is_foreach()
            and len(node1.get_nodes()) + len(node2.get_nodes()) > config.max_fusion_size
        ):
            WhyNoFuse(node1, node2)("exceeds max fusion")
            return False  # heuristic not needed for correctness

        if V.choices.can_fusion_increase_peak_memory(scheduler, node1, node2):
            WhyNoFuse(node1, node2)("Fusion will increase peak memory")
            return False

        return True

    @staticmethod
    def can_fuse_vertical(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """Hook for heuristics to prevent vertical (producer/consumer) fusions"""
        return True

    @staticmethod
    def can_fuse_horizontal(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
        shared_data_score: int,
    ) -> bool:
        """Hook for heuristics to prevent horizontal (consumer/consumer) fusions"""
        if shared_data_score < config.score_fusion_memory_threshold:
            WhyNoFuse(node1, node2)("score_fusion_memory_threshold")
            return False
        if V.choices.are_long_distant_nodes(node1, node2):
            WhyNoFuse(node1, node2)(
                "Nodes are too far away. Fusing them may increase peak memory."
            )
            return False
        return True

    @staticmethod
    def score_fusion(
        scheduler: Scheduler,
        node1: BaseSchedulerNode,
        node2: BaseSchedulerNode,
    ) -> Sortable:
        """
        Assign a score (higher comes first) to the fusion of node1 and node2.
        When different fusions conflict with each other, this is the way we
        decide what order to run them in.

        Our current score is based on:
        - The type of fusion (template/reduction/etc)
        - Estimate of the saved memory operations
        - Fusions closer together in original graph order
        """
        memory_score = scheduler.score_fusion_memory(node1, node2)
        proximity_score = -max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )

        # prologue fusion always last
        if node2.is_template():
            template_score = 0
        else:
            template_score = 1 + (
                (node1.is_template() == config.epilogue_fusion_first)
                and memory_score > 0
            )

        return (
            template_score,
            node1.is_reduction() == node2.is_reduction() and memory_score > 0,
            memory_score,
            proximity_score,
        )

    @staticmethod
    def estimate_runtime(
        counted_flops: int,
        counted_bytes: int,
        gpu_flops: int,
        gpu_memory_bandwidth: int,
    ) -> float:
        """
        Return estimated runtime in nanoseconds.
        """
        # TODO(xmfan): find a better heuristic to model FLOPS/latency relationship
        factor = 1.0
        compute_time = (factor * counted_flops / gpu_flops) * 1e9
        transfer_time = counted_bytes / gpu_memory_bandwidth

        # Return estimated runtime in nanoseconds
        return max(compute_time, transfer_time)

    @staticmethod
    def pick_loop_order(
        stride_lengths: list[list[int]],
        sizes: Sequence[sympy.Expr],
        priority_idx: tuple[int, ...] = (),
    ) -> list[int]:
        """
        A heuristic to decide loop iteration orders.  This has not been well
        tuned and may be something we should autotune.
        """

        @functools.cmp_to_key
        def index_cmp(a: int, b: int) -> int:
            if sizes[a] == 1 or sizes[b] == 1:
                # 1-sizes don't matter, just move them to the end
                return cmp(sizes[a] == 1, sizes[b] == 1)

            # Take abs, otherwise flipped dimensions are treated as smaller
            # strides than contiguous dims
            stride_len_a = [abs(sl[a]) for sl in stride_lengths]
            stride_len_b = [abs(sl[b]) for sl in stride_lengths]

            # equivalent to
            # np.logical_or(stride_lengths[:, b] == 0, stride_lengths[:, a] < stride_lengths[:, b]).all()
            a_first = sum(
                sl_b == 0 or sl_a < sl_b
                for sl_a, sl_b in zip(stride_len_a, stride_len_b)
            )
            b_first = sum(
                sl_a == 0 or sl_b < sl_a
                for sl_a, sl_b in zip(stride_len_a, stride_len_b)
            )
            if a_first > b_first:
                return -1
            if b_first > a_first:
                return 1

            # otherwise contiguous
            return cmp(b, a)

        order = list(reversed(range(len(stride_lengths[0]))))
        if len(priority_idx) > 0:
            # if we have priority node, only use that node's order
            stride_lengths = [stride_lengths[pi] for pi in priority_idx]
        if config.pick_loop_orders:
            order.sort(key=index_cmp)
        return order

    @staticmethod
    def can_fusion_increase_peak_memory(
        scheduler: Scheduler, node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        Return true if fusing the two nodes can potentially increasing peak memory.

        The implementation is more like a heuristic since we don't really know if we are at peak
        or not when trying to fuse these two ndoes. The order of nodes may change later which makes the
        peak memory estimation hard.

        Here is how we decide the LOWER BOUND of extra memory allocation if we fuse these 2 nodes:
        1. find all buffers read by each node with a single user. These buffers are supposed to
           be reused if we don't fuses these 2 nodes
        2. find the intersection of these buffers for the two node and sum the total buffer size.
           If we don't fuse these two nodes, we can at lease avoid this much memory allocation.
           Note that the extra memory allocation is not necessarily causing peak memory increase.
           This is just a heuristic.

        We return true only if the saving for fusion can not trade off the extra memory allocation.
        """

        from .codegen.wrapper import buffer_reuse_key

        def _find_single_user_inputs(
            node: BaseSchedulerNode,
        ) -> list[ir.Buffer]:
            output = []
            for rd in node.read_writes.reads:
                buf = scheduler.name_to_buf.get(rd.name)
                if buf and len(buf.users) == 1 and buf.node.has_tensor_output():
                    output.append(buf.node)
            return output

        # Check inputs that can be potentially reused
        lhs_dep_nodes = _find_single_user_inputs(node1)
        rhs_dep_nodes = _find_single_user_inputs(node2)

        lhs_reuse_keys = OrderedSet(buffer_reuse_key(buf) for buf in lhs_dep_nodes)
        rhs_reuse_keys = OrderedSet(buffer_reuse_key(buf) for buf in rhs_dep_nodes)

        common_reuse_keys = lhs_reuse_keys.intersection(rhs_reuse_keys)

        memory_overhead = 0
        for key in common_reuse_keys:
            try:
                memory_overhead += int(key[2])
            except ValueError:
                # not an interger. Fallback is to fuse
                return False

        bw_saving = scheduler.score_fusion_memory(node1, node2)

        # The factor 32 here is quite arbitrary.
        if V.graph.sizevars.statically_known_gt(memory_overhead, 32 * bw_saving):
            return True
        return False

    @staticmethod
    def are_long_distant_nodes(
        node1: BaseSchedulerNode, node2: BaseSchedulerNode
    ) -> bool:
        """
        This function prevents fusion for nodes that can increase memory
        footprint. This problem is more common in horizontal fusion, where nodes
        that are far apart in the original order get fused, lengthening the live
        intervals of tensors. This is very evident in models with activation
        checkpointing, where the recomputed nodes from different checkpointed
        regions get fused and significantly increase the memory footprint.

        The current attempt is a quick, possibly hacky, heuristic to prevent the
        fusion of nodes that are far away in the original order.

        A better but difficult to implement heurisitic would be to use live
        intervals of the buffers, find region of peak pressure in the original
        program and prevent fusion that crosses that peak region. We might need
        special care or good approximation in this implementation, as fusion of
        node changes live intervals, and re-computing live intervals and peak
        memory after each fusion can introduce large compilation overhead.
        """
        proximity_score = max(
            abs(node1.min_order - node2.max_order),
            abs(node2.min_order - node1.max_order),
        )
        return proximity_score > 64

    @staticmethod
    def check_prologue_fusion_heuristics_fusable(
        prologue_node: BaseSchedulerNode,
        template_node: BaseSchedulerNode,
        why: WhyNoFuse,
    ) -> bool:
        """
        Heuristics to avoid benchmarking predictably slow prologue fusions
        """
        # user opt into more aggressive prologue fusion, dont use heuristics
        if prologue_node.get_operation_names() <= V.graph.invoke_quant_ops:
            return True

        read_bytes = prologue_node.get_read_buffer_sizes()
        write_bytes = prologue_node.get_write_buffer_sizes()

        # Initially, only do fusions which will result in fewer memory accesses inside of the template to avoid
        # potential bad cache behavior and shared memory use.
        # we also want to avoid benchmarking reliably unprofitable fusions like downcasts from fp32 -> fp16 inside kernel.
        # allowing gathers by allowing increasing write_bytes by small factor
        # TODO - make configurable per input, for insance, bias can fuse fp32 -> fp16 profitably

        BYTES_THRESHOLD_MULTIPLIER = 1.1
        if read_bytes > (write_bytes * BYTES_THRESHOLD_MULTIPLIER):
            why("prologue fusion will not increase amount of bytes read in kernel")
            return False

        # we want to avoid attempting to fuse predictably unprofitable prologues
        # such as increasing the unaligned reads or writes.
        # TODO - would be nice to generalize this, however, we would need more explicit
        # knowledge of memory access patterns in the TritonTemplate in order to know
        # the stride order to check alignment.
        origins = tuple(
            e.target
            for n in prologue_node.get_nodes()
            if n.node is not None
            for e in n.node.get_origins()
            if e.op == "call_function"
        )
        if origins == (torch.ops.aten.constant_pad_nd.default,):
            why(
                "prologue fusion will not increase attempt to fuse in padding bc it increases unaligned reads"
            )
            return False

        def low_prec_fp(dtype: torch.dtype) -> bool:
            return dtype.itemsize <= 2 and dtype.is_floating_point

        if (
            low_prec_fp(template_node.get_template_node_or_throw().dtype)
            and not prologue_node.can_codegen_in_low_precision()
        ):
            why(
                "prologue fusion that must be upcast to fp32 not profitable for low precision templates"
            )
            return False

        return True

    @staticmethod
    def decide_layout_opt(gm: GraphModule, *, is_inference: bool) -> bool:
        """
        Decide if we should enable layout optimization for this graph based on
        heuristics.
        """
        if not config.layout_optimization:
            return False

        if config.force_layout_optimization:
            return True

        conv_nodes = [
            n for n in gm.graph.nodes if n.target == torch.ops.aten.convolution.default
        ]
        nconv = len(conv_nodes)

        if nconv == 0:
            return False

        # For cpu backend and mkldnn enabled, we always use channels_last for better performance.
        if (
            torch.backends.mkldnn.enabled
            and torch.backends.mkldnn.is_available()
            and all(
                n.args[idx].meta["val"].device == torch.device("cpu")
                for n in conv_nodes
                for idx in [0, 1]
            )
        ):
            return True

        # Following models are skipped due to this:
        # jx_nest_base
        # volo_d1_224
        if len(list(gm.graph.nodes)) >= 300 * nconv:
            log.debug("Skipped layout opt because only a few conv")
            return False

        if any(
            has_free_symbols(n.args[idx].meta["val"])
            for n in conv_nodes
            for idx in [0, 1]
        ):
            log.debug(
                "See perf regression with dynamic shape. Follow up in https://github.com/pytorch/pytorch/issues/102670"
            )
            return False

        def is_grouped(n: Any) -> bool:
            meta_val = n.args[1].meta["val"]  # type: ignore[union-attr, operator]
            assert isinstance(meta_val, torch.Tensor)
            return n.args[-1] > 1 and meta_val.size(1) > 1  # type: ignore[union-attr, operator]

        def is_in_out_channel(n: torch.fx.Node) -> bool:
            return (
                n.args[1].meta["val"].size(0) * 2 <= n.args[1].meta["val"].size(1)  # type: ignore[union-attr, operator]
                and n.args[1].meta["val"].size(2) > 1  # type: ignore[union-attr, operator]
            )

        def is_small_channel(n: torch.fx.Node) -> bool:
            return (
                n.args[1].meta["val"].size(0) <= 64  # type: ignore[union-attr, operator]
                and n.args[1].meta["val"].size(1) <= 64  # type: ignore[union-attr, operator]
            )

        # only grouped convolutions benchmarked as slower in conv samples for inference only
        if is_inference:
            from torch.utils.flop_counter import FlopCounterMode

            flop_counts: dict[str, float] = defaultdict(float)
            for node in conv_nodes:
                success, args, kwargs = torch._inductor.fx_utils.get_fake_args_kwargs(
                    node
                )

                if success:
                    with FlopCounterMode(display=False) as flop_counter_mode:
                        with V.fake_mode:
                            node.target(*args, **kwargs)

                    counted_flops = flop_counter_mode.get_total_flops()
                    if is_grouped(node):
                        node_type = "grouped"
                    elif is_small_channel(node):
                        node_type = "small"
                    elif is_in_out_channel(node):
                        node_type = "in_out"
                    else:
                        node_type = "default"

                    flop_counts[node_type] += counted_flops
                else:
                    log.debug("Conv inputs meta not found")

            # average benchmarked channels last speedup / slowdown, < 1 is speedup.
            # taken from the set of convolution inputs in benchmarks/dynamo/microbenchmarks/operator_inp_logs/torchbench_train/
            # To regenerate these numbers follow https://gist.github.com/eellison/55d7a6ed6f39829d68ac56f95f4df5bb
            GROUPED_MULTIPLIER = 1.358
            DEFAULT_MULTIPLIER = 0.823
            IN_OUT_MULTIPLIER = 0.725
            SMALL_MULTIPLIER = 0.783

            total_flops = sum(flop_counts.values())
            # TODO - get different values per hardware
            weighted_flops = (
                flop_counts["grouped"] * GROUPED_MULTIPLIER
                + flop_counts["small"] * SMALL_MULTIPLIER
                + flop_counts["in_out"] * IN_OUT_MULTIPLIER
                + flop_counts["default"] * DEFAULT_MULTIPLIER
            )
            do_layout_opt = weighted_flops <= total_flops
            if not do_layout_opt:
                log.debug(
                    "Skipped layout opt in inference because weighted flops indicate slowdown, default: %d, channels last: %d",
                    total_flops,
                    weighted_flops,
                )
            return do_layout_opt

        # Channels last layout can dramatically hurt grouped conv perf. E.g.
        # Conv with arguments like
        #   {"input_shape": [32, 224, 112, 112], "weight_shape": [224, 112, 3, 3],
        #    "stride": [2, 2], "padding": [1, 1], "groups": 2}
        # slows down 31x using channels last..

        # But a lot of timm models use depthwise separable convolution which will
        # result in grouped convolution with in-channel size == 1.
        # For those grouped convolution, channels last still helps a lot.
        # E.g.
        # Conv with arguments
        #   {"input_shape": [128, 58, 56, 56], "weight_shape": [58, 1, 3, 3],
        #    "stride": [2, 2], "padding": [1, 1], "groups": 58}
        # get 1.86x speedup with channels last layout.
        #
        # The following heuristics skip using channels-last if the model contains
        # grouped convolution with in-channels > 1.
        if any(map(is_grouped, conv_nodes)):
            log.debug(
                "Skip layout opt because found grouped convolution with >1 in_channels!"
            )
            return False

        # For some models that contain convolution with larger in-channel than out-channel, applying
        # channels last hurts performance.
        # Following models are skipped due to this:
        # - pytorch_unet
        # - phlippe_densenet (slightly worse)
        # - Background_Matting (1.22x -> 0.821x)
        # - pytorch_CycleGAN_and_pix2pix (1.597x -> 1.294x)
        if any(map(is_in_out_channel, conv_nodes)):
            log.debug(
                "Skip layout opt because some convolutions have smaller out_channel"
            )
            return False

        # Following models are skipped due to this:
        # - functorch_maml_omniglot
        if all(map(is_small_channel, conv_nodes)):
            log.debug("Skip layout opt because all convolution channels are too small")
            return False

        return True

    @staticmethod
    def reorder_for_peak_memory(
        nodes: list[BaseSchedulerNode],
        name_to_buf: dict[str, SchedulerBuffer],
        name_to_fused_node: dict[str, BaseSchedulerNode],
        graph_inputs: OrderedSet[str],
        graph_outputs: OrderedSet[str],
        methods: list[Callable[..., list[BaseSchedulerNode]]] = [  # noqa: B006
            topological_sort_lpmf,
            topological_sort_bfs,
            topological_sort_dfs,
        ],
    ) -> list[BaseSchedulerNode]:
        """
        Try a few heuristics based topological sort algorithms, and pick the one whose
        resulting topological order has the lowest peak memory estimation.
        """

        return reorder_for_peak_memory(
            nodes, name_to_buf, name_to_fused_node, graph_inputs, graph_outputs, methods
        )

    @staticmethod
    def should_realize_on_reuse(box: StorageBox, users):  # type: ignore[no-untyped-def]
        """
        A heuristic to decide if we should realize a tensor
        that is used multiple times.
        """
        if users > 1 and isinstance(box.data, (Pointwise, Reduction)):
            if is_cpu(box.data):
                # Heuristic for realizing reused result of heavy ops on cpu
                opcount = box.data.inner_fn_opcount()
                heavy_ops = ["exp", "sigmoid"]  # a list of heavy ops
                if any(x in opcount.used_ops for x in heavy_ops):
                    return True
            return (
                box.num_reads() > config.realize_reads_threshold
                or box.has_large_inner_fn()
            )
        return False

    @staticmethod
    def pad_mm_get_alignment_size_dtype(dtype: torch.dtype) -> int:
        """
        This heuristic determines the mapping from dtype to the multiple of which matmul dimensions are padded.

        This padding is applied to all dimensions in a padded matmul. See pad_mm for usage.
        """
        if dtype == torch.float16 or dtype == torch.half or dtype == torch.bfloat16:
            return 8
        elif dtype == torch.float32 or dtype == torch.float:
            return 4
        else:
            return 0
