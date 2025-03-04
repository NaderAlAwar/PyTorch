# mypy: allow-untyped-defs
import operator
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, cast, Optional

import torch
from torch.utils._ordered_set import OrderedSet

from .. import config, inductor_prims
from ..pattern_matcher import (
    CallFunction,
    Ignored,
    KeywordArg,
    ListOf,
    Match,
    MULTIPLE,
    PatternExpr,
    PatternMatcherPass,
)


aten = torch.ops.aten
patterns = PatternMatcherPass()


def _is_backward(graph: torch.fx.Graph) -> bool:
    placeholders = []
    for node in graph.nodes:
        if node.op != "placeholder":
            break
        placeholders.append(node)
    return not all(node.name.startswith("primal") for node in placeholders)


def _compute_mm_arithmetic_intensity(M: int, N: int, K: int) -> float:
    return M * N * K / (M * K + N * K + M * N)


def _filter_nodes_by_target(nodes: list[torch.fx.Node], target) -> list[torch.fx.Node]:
    return [x for x in nodes if x.target == target]


def _find_ancestors(node: torch.fx.Node) -> OrderedSet[torch.fx.Node]:
    ancestors = OrderedSet[torch.fx.Node]()
    ancestors.add(node)
    cur_nodes = [node]
    while len(cur_nodes) > 0:
        new_nodes = []
        for node in cur_nodes:
            for inp in node.all_input_nodes:
                if inp not in ancestors:
                    ancestors.add(inp)
                    new_nodes.append(inp)
        cur_nodes = new_nodes
    return OrderedSet([node for node in ancestors if node.op != "placeholder"])


def _get_tensor(node: torch.fx.Node) -> torch.Tensor:
    val = node.meta["val"]
    assert isinstance(val, torch.Tensor)
    return val


@dataclass
class _AllGatherMatch:
    match: Match
    shard_node: torch.fx.Node
    ag_node: torch.fx.Node
    res_node: torch.fx.Node
    gather_dim: int
    group_name: str

    def replace_with(self, new_node: torch.fx.Node) -> None:
        self.res_node.replace_all_uses_with(new_node)

    def erase(self) -> None:
        for node in reversed(self.match.nodes):
            if len(node.users) == 0:
                node.graph.erase_node(node)


def find_all_gather_patterns(graph: torch.fx.Graph):
    c10d = torch.ops._c10d_functional

    def make_zero_dim_all_gather_pattern(shard):
        return CallFunction(
            c10d.wait_tensor.default,
            CallFunction(
                c10d.all_gather_into_tensor.default,
                shard,
                Ignored(),
                KeywordArg("group_name"),
            ),
        )

    # Matches funcol.all_gather_tensor with gather_dim == 0
    zero_dim_all_gather_pattern = make_zero_dim_all_gather_pattern(KeywordArg("shard"))

    def make_all_gather_split_pattern(shard):
        return CallFunction(
            operator.getitem,
            CallFunction(
                aten.split.Tensor,
                make_zero_dim_all_gather_pattern(shard),
                Ignored(),
                _users=MULTIPLE,
            ),
            Ignored(),
        )

    def make_cat_pattern(splits):
        return CallFunction(
            aten.cat.default,
            ListOf(splits),
            KeywordArg("gather_dim"),
        )

    # Matches funcol.all_gather_tensor with gather_dim > 0
    non_zero_dim_all_gather_pattern = make_cat_pattern(
        make_all_gather_split_pattern(KeywordArg("shard")),
    )

    # Match a zero-dim all-gather in which the data is transferred as uint8 and
    # viewed back as the original dtype.
    zero_dim_type_erased_all_gather_pattern = CallFunction(
        aten.view.dtype,
        make_zero_dim_all_gather_pattern(
            KeywordArg("shard"),
        ),
        Ignored(),
    )

    # Match a non-zero dim all-gather in which the data is transferred as uint8
    # and viewed back as the original dtype.
    non_zero_dim_type_erased_all_gather_pattern = CallFunction(
        aten.view.dtype,
        make_cat_pattern(
            CallFunction(
                aten.view.dtype,
                make_all_gather_split_pattern(
                    KeywordArg("shard"),
                ),
                Ignored(),
            ),
        ),
        Ignored(),
    )

    # If two patterns with the same res_node_target have the same suffix, the
    # longer pattern should appear first in the list.
    # e.g. supposed we have (1) A -> B -> C -> D and (2) B -> C -> D, (1)
    # should appear before (2) in the list.
    res_node_target_to_patterns = {
        aten.cat.default: [
            (non_zero_dim_all_gather_pattern, 0),
        ],
        aten.view.dtype: [
            (non_zero_dim_type_erased_all_gather_pattern, 0),
            (zero_dim_type_erased_all_gather_pattern, 0),
        ],
        c10d.wait_tensor.default: [
            (zero_dim_all_gather_pattern, 0),
        ],
    }

    # Match in reverse to ensure longer patterns is prioritized
    all_gathers = []
    visited_ag_nodes = OrderedSet[torch.fx.Node]()
    for node in reversed(graph.nodes):
        for target, patterns in res_node_target_to_patterns.items():
            if node.target != target:
                continue
            for pattern, ag_node_idx in patterns:
                match = pattern.match(node)
                if not match:
                    continue

                assert isinstance(match, Match)
                ag_node = match.nodes[ag_node_idx]
                assert ag_node.target == c10d.all_gather_into_tensor.default

                if ag_node in visited_ag_nodes:
                    continue
                visited_ag_nodes.add(ag_node)

                ag_match = _AllGatherMatch(
                    match=match,
                    shard_node=match.kwargs["shard"],
                    ag_node=ag_node,
                    res_node=node,
                    gather_dim=match.kwargs.get("gather_dim", 0),
                    group_name=match.kwargs["group_name"],
                )
                all_gathers.append(ag_match)

    return [*reversed(all_gathers)]


@dataclass
class _ReduceScatterMatch:
    match: Match
    input_node: torch.fx.Node
    rs_node: torch.fx.Node
    res_node: torch.fx.Node
    reduce_op: str
    scatter_dim: int
    group_name: str

    def replace_with(self, new_node: torch.fx.Node) -> None:
        self.res_node.replace_all_uses_with(new_node)

    def erase(self) -> None:
        for node in reversed(self.match.nodes):
            if len(node.users) == 0:
                node.graph.erase_node(node)


def find_reduce_scatter_patterns(graph: torch.fx.Graph):
    c10d = torch.ops._c10d_functional

    def reduce_scatter_template(inp: PatternExpr):
        return CallFunction(
            c10d.wait_tensor.default,
            CallFunction(
                c10d.reduce_scatter_tensor.default,
                inp,
                KeywordArg("reduce_op"),
                Ignored(),
                KeywordArg("group_name"),
            ),
        )

    # Matches funcol.reduce_scatter_tensor with scatter_dim == 0
    zero_dim_reduce_scatter_pattern = reduce_scatter_template(KeywordArg("input"))

    # Matches funcol.reduce_scatter_tensor with scatter_dim > 0
    non_zero_dim_reduce_scatter_pattern = reduce_scatter_template(
        CallFunction(
            aten.cat.default,
            ListOf(
                CallFunction(
                    operator.getitem,
                    CallFunction(
                        aten.split.Tensor,
                        KeywordArg("input"),
                        Ignored(),
                        KeywordArg("scatter_dim"),
                        _users=MULTIPLE,
                    ),
                    Ignored(),
                )
            ),
        ),
    )

    reduce_scatters = []
    for node in reversed(graph.nodes):
        if node.target == c10d.wait_tensor.default:
            if match := non_zero_dim_reduce_scatter_pattern.match(node):
                assert isinstance(match, Match)
                reduce_scatters.append(
                    _ReduceScatterMatch(
                        match=match,
                        input_node=match.kwargs["input"],
                        rs_node=match.nodes[-2],
                        res_node=node,
                        reduce_op=match.kwargs["reduce_op"],
                        scatter_dim=match.kwargs["scatter_dim"],
                        group_name=match.kwargs["group_name"],
                    )
                )
            elif match := zero_dim_reduce_scatter_pattern.match(node):
                assert isinstance(match, Match)
                reduce_scatters.append(
                    _ReduceScatterMatch(
                        match=match,
                        input_node=match.kwargs["input"],
                        rs_node=match.nodes[0],
                        res_node=node,
                        reduce_op=match.kwargs["reduce_op"],
                        scatter_dim=0,
                        group_name=match.kwargs["group_name"],
                    )
                )
    return [*reversed(reduce_scatters)]


@dataclass
class _Matmul:
    nodes: list[torch.fx.Node]
    arg_ancestor_nodes: OrderedSet[torch.fx.Node] = field(init=False)
    A_node: torch.fx.Node
    B_node: torch.fx.Node

    def __post_init__(self):
        assert len(self.nodes) in (1, 3)
        if len(self.nodes) == 1:
            assert self.nodes[0].target in (aten.mm.default, aten._scaled_mm.default)
        else:
            assert self.nodes[0].target == aten.reshape.default
            assert self.nodes[1].target in (aten.mm.default, aten._scaled_mm.default)
            assert self.nodes[2].target == aten.reshape.default
        self.arg_ancestor_nodes = _find_ancestors(self.B_node)

    def replace_with(self, new_node: torch.fx.Node) -> None:
        """
        Replace the matmul with the new node.
        """
        graph = new_node.graph

        # For 2D-matmuls, we simply replace the mm node with `new_node`.
        if len(self.nodes) == 1:
            mm_node = self.nodes[0]
            assert mm_node.target in (aten.mm.default, aten._scaled_mm.default)
            mm_node.replace_all_uses_with(new_node)
            graph.erase_node(mm_node)
            return

        # An ND-matmul is reshape -> mm -> reshape sequence. We first replace
        # the second reshape node with `new_node`. Then, we ensure that the
        # original mm node in the sequence ends up with zero users by replacing
        # it with a reverse reshape of `new_node`.
        graph = new_node.graph
        assert len(self.nodes) == 3
        mm_node = self.nodes[1]
        output_reshape_node = self.nodes[2]

        assert mm_node.target in (aten.mm.default, aten._scaled_mm.default)
        assert output_reshape_node.target == aten.reshape.default

        output_reshape_node.replace_all_uses_with(new_node)
        if len(mm_node.users) > 1:
            with graph.inserting_after(new_node):
                new_mm_node = graph.call_function(
                    aten.reshape.default,
                    args=(new_node, [*_get_tensor(mm_node).shape]),
                )
            mm_node.replace_all_uses_with(new_mm_node)

    def erase(self) -> None:
        for node in reversed(self.nodes):
            if len(node.users) == 0:
                node.graph.erase_node(node)

    @classmethod
    def from_match(cls, match: list[torch.fx.Node]) -> "_Matmul":
        assert len(match) in (1, 3)
        assert match[0].target in (
            aten.mm.default,
            aten.reshape.default,
        )
        mm_node = match[0] if len(match) == 1 else match[1]
        return _Matmul(
            nodes=match,
            A_node=cast(torch.fx.Node, match[0].args[0]),
            B_node=cast(torch.fx.Node, mm_node.args[1]),
        )


@dataclass
class _ScaledMatmul(_Matmul):
    A_scale_node: torch.fx.Node
    B_scale_node: torch.fx.Node
    bias_node: Optional[torch.fx.Node]
    result_scale_node: Optional[torch.fx.Node]
    out_dtype: Optional[torch.dtype]
    use_fast_accum: bool

    def __post_init__(self):
        super().__post_init__()
        self.arg_ancestor_nodes |= _find_ancestors(self.A_scale_node)
        self.arg_ancestor_nodes |= _find_ancestors(self.B_scale_node)

    @classmethod
    def from_match(cls, match: list[torch.fx.Node]) -> "_ScaledMatmul":
        assert len(match) in (1, 3)
        assert match[0].target in (
            aten._scaled_mm.default,
            aten.reshape.default,
        )
        mm_node = match[0] if len(match) == 1 else match[1]

        def get_arg(node: torch.fx.Node, idx: int, default: Any) -> Any:
            if idx >= len(node.args):
                return default
            return node.args[idx]

        def insert_reshape_op(node: torch.fx.Node):
            """
            Given a reciprocal node with a parent reshape node,
            insert a reshape node after the reciprocal node which reshapes
            the reciprocal output back to the original shape before the first reshape.

            Before:
                reshape (a,bc,) to (a*b,c) -> reciprocal

            After:
                reshape (a,bc,) to (a*b,c) -> reciprocal -> reshape (a*b,c) to (a,b,c)

            Returns the new reshape node.
            """
            # ensure the given node matches the pattern described in the docstring
            assert node.target == aten.reciprocal.default, (
                "Node must be a aten.reciprocal.default op"
            )
            assert len(node.all_input_nodes) == 1, "Node must have exactly one parent"

            parent_node = node.all_input_nodes[0]
            assert parent_node.target == aten.reshape.default, (
                "Parent node must be a aten.reshape.default op"
            )
            assert len(parent_node.all_input_nodes) == 1, (
                "Parent node must have exactly one input node"
            )

            parent_input_node = parent_node.all_input_nodes[0]
            parent_input_shape = [*_get_tensor(parent_input_node).shape]

            # insert reshape back to shape from before the parent reshape op
            graph = node.graph
            with graph.inserting_after(node):
                reshape_node = graph.call_function(
                    aten.reshape.default, (node, parent_input_shape)
                )

            # ensure all users of original node (except the reshape node) now use the reshaped node instead
            node_users = [*node.users]
            for user in node_users:
                if user != reshape_node:
                    user.replace_input_with(node, reshape_node)

            return reshape_node

        is_reshape_mm_reshape_pattern = match[0].target == aten.reshape.default
        mm_node = match[1] if is_reshape_mm_reshape_pattern else match[0]

        # `A_node` is pulled directly from match rather than `mm_node` because it needs to handle
        # both of the following cases:
        #
        # Case 1: single node match (mm):
        # - match[0].args[0] will be the "A tensor" node of scaled_mm
        # - Has 2D shape
        #
        # Case 2: 3 node match (reshape -> mm -> reshape)
        # - match[0].args[0] will be the "A tensor" input to the reshape op
        # - Has 3D+ shape
        A_node = cast(torch.fx.Node, match[0].args[0])
        B_node = cast(torch.fx.Node, mm_node.args[1])
        A_scale_node = cast(torch.fx.Node, mm_node.args[2])
        B_scale_node = cast(torch.fx.Node, mm_node.args[3])

        A_ndim = _get_tensor(A_node).ndim
        A_scale_ndim = _get_tensor(A_scale_node).ndim
        is_reciprocal_with_reshape_parent = (
            A_scale_node.target == aten.reciprocal.default
            and len(A_scale_node.all_input_nodes) == 1
            and A_scale_node.all_input_nodes[0].target == aten.reshape.default
        )
        is_tensorwise_scaling = A_scale_ndim <= 1

        # This is a temporary workaround to handle the reshape -> scaled_mm -> reshape
        # pattern when scales are row-wise, and have been reshaped along with the target
        # tensor. See https://github.com/pytorch/pytorch/pull/148001 for details.
        #
        # If tensor dim does not match scale dim, check if the scale node follows
        # the "reshape -> reciprocal" pattern. If so, we can insert a reshape op after
        # the reciprocal, to reshape the reciprocal back to the original shape before
        # the first reshape op.
        #
        # TODO: remove this workaround once torch._scaled_matmul exists and can be used
        # to implement a more robust long-term support for 3D+ scaled matmuls.
        if (
            is_reshape_mm_reshape_pattern
            and A_ndim != A_scale_ndim
            and not is_tensorwise_scaling
            and is_reciprocal_with_reshape_parent
        ):
            A_scale_node = insert_reshape_op(A_scale_node)

        return _ScaledMatmul(
            nodes=match,
            A_node=A_node,
            B_node=B_node,
            A_scale_node=A_scale_node,
            B_scale_node=B_scale_node,
            bias_node=get_arg(mm_node, 4, None),
            result_scale_node=get_arg(mm_node, 5, None),
            out_dtype=get_arg(mm_node, 6, None),
            use_fast_accum=get_arg(mm_node, 7, False),
        )


def _find_reshape_mm_reshape(node: torch.fx.Node) -> list[_Matmul]:
    if node.target != aten.reshape.default:
        return []

    matches = []
    for mm_node in node.users:
        if mm_node.target not in (aten.mm.default, aten._scaled_mm.default):
            continue
        for reshape_node in mm_node.users:
            if reshape_node.target != aten.reshape.default:
                continue

            # Since the reshape -> mm -> reshape pattern would be subsumed into
            # the fused op, we only match the patterns where the shape of the
            # second reshape is matches the mm result produced by the fused op.
            matmul_input_node = cast(torch.fx.Node, node.args[0])
            B_node = cast(torch.fx.Node, mm_node.args[1])
            matmul_out_shape = torch.Size(
                [
                    *_get_tensor(matmul_input_node).shape[:-1],
                    _get_tensor(B_node).shape[-1],
                ]
            )
            if _get_tensor(reshape_node).shape != matmul_out_shape:
                continue
            matches.append([node, mm_node, reshape_node])
            # If for some rare reason mm_node is being reshaped by two
            # different reshape nodes, we only include mm_node once in the
            # parsing result.
            break

    matmuls = []
    for match in matches:
        mm_node = match[1]
        if mm_node.target == aten.mm.default:
            matmul = _Matmul.from_match(match)
            matmuls.append(matmul)
        elif mm_node.target == aten._scaled_mm.default:
            matmul = _ScaledMatmul.from_match(match)
            matmuls.append(matmul)
        else:
            raise AssertionError(
                "Expect the node's target to be either aten.mm.default or "
                f"aten._scaled_mm.default. Got {mm_node.target}."
            )
    return matmuls


def _find_consumer_matmuls(node: torch.fx.Node) -> list[_Matmul]:
    """
    Find the matmuls that use `node` as the lhs argument.
    """
    matmuls = []
    for user in node.users:
        # ND matmuls
        if user.target == aten.reshape.default:
            matmuls.extend(_find_reshape_mm_reshape(user))
        # 2D matmuls
        elif user.target == aten.mm.default:
            matmul = _Matmul.from_match(match=[user])
            matmuls.append(matmul)
        elif user.target == aten._scaled_mm.default:
            matmul = _ScaledMatmul.from_match([user])
            matmuls.append(matmul)
    return matmuls


def _insert_fused_all_gather_matmul(
    graph: torch.fx.Graph,
    matmuls: list[_Matmul],
    shard_node: torch.fx.Node,
    gather_dim: int,
    group_name: str,
) -> torch.fx.Node:
    mm_types = OrderedSet(map(type, matmuls))
    assert len(mm_types) == 1
    mm_type = next(iter(mm_types))
    if mm_type == _Matmul:
        B_nodes = [matmul.B_node for matmul in matmuls]
        return graph.call_function(
            torch.ops.symm_mem.fused_all_gather_matmul.default,
            args=(shard_node, B_nodes, gather_dim, group_name),
            kwargs={"return_A": True},
        )
    elif mm_type == _ScaledMatmul:
        scaled_matmuls = cast(list[_ScaledMatmul], matmuls)
        return graph.call_function(
            torch.ops.symm_mem.fused_all_gather_scaled_matmul.default,
            args=(
                shard_node,
                [matmul.B_node for matmul in scaled_matmuls],
                scaled_matmuls[0].A_scale_node,
                [matmul.B_scale_node for matmul in scaled_matmuls],
                gather_dim,
                group_name,
                [matmul.bias_node for matmul in scaled_matmuls],
                [matmul.result_scale_node for matmul in scaled_matmuls],
                [matmul.out_dtype for matmul in scaled_matmuls],
                [matmul.use_fast_accum for matmul in scaled_matmuls],
            ),
        )
    else:
        raise AssertionError(f"Unexpected matmul match type: {mm_type}")


def fuse_all_gather_matmul(all_gather: _AllGatherMatch) -> None:
    """
    Fused the pattern

        A = all_gather_tensor(A_shard, gather_dim, group_name)
        C_0 = torch.matmul(A, B_0)
        C_1 = torch.matmul(A, B_1)
        C_2 = torch.matmul(A, B_2)
        ...

    into

        A, Cs = torch.ops.symm_mem.fused_all_gather_matmul(
            A_shard, [B_0, B_1, B_2, ...], gather_dim, group_name,
        )
    """
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        return

    from torch.distributed._symmetric_memory import (
        is_symm_mem_enabled_for_group,
        restride_A_shard_for_fused_all_gather_matmul,
    )

    shard_node, ag_node, ag_res_node, gather_dim, group_name = (
        all_gather.shard_node,
        all_gather.ag_node,
        all_gather.res_node,
        all_gather.gather_dim,
        all_gather.group_name,
    )

    if not is_symm_mem_enabled_for_group(group_name):
        return

    if gather_dim >= len(_get_tensor(shard_node).shape) - 1:
        # Decomposing the matmul on the K dimension is not supported
        return

    # Find consumer matmuls
    matmuls = _find_consumer_matmuls(ag_res_node)

    # The matmuls are only fusible if non-A args don't depend on the all-gather
    # result node
    matmuls = [
        matmul
        for matmul in matmuls
        if all_gather.res_node not in matmul.arg_ancestor_nodes
    ]

    if len(matmuls) == 0 or len(OrderedSet(map(type, matmuls))) != 1:
        return

    # Fuse the all_gather_tensor with the eligible matmuls
    graph = ag_node.graph
    with graph.inserting_before(ag_node):
        if "val" in shard_node.meta:
            restrided = restride_A_shard_for_fused_all_gather_matmul(
                _get_tensor(shard_node),
                gather_dim,
            )
            shard_node = graph.call_function(
                inductor_prims.force_stride_order,
                args=(shard_node, restrided.stride()),
            )

        fused_node = _insert_fused_all_gather_matmul(
            graph, matmuls, shard_node, gather_dim, group_name
        )
        new_ag_node = graph.call_function(
            operator.getitem,
            args=(fused_node, 0),
        )
        new_out_nodes = graph.call_function(
            operator.getitem,
            args=(fused_node, 1),
        )
        for idx, matmul in enumerate(matmuls):
            new_out_node = graph.call_function(
                operator.getitem,
                args=(new_out_nodes, idx),
            )
            matmul.replace_with(new_out_node)
            matmul.erase()
        all_gather.replace_with(new_ag_node)
        all_gather.erase()

        # If the new_ag_node has no users, we tell the fused op to not return
        # it. This creates more optimization opportunities.
        if len(new_ag_node.users) == 0:
            graph.erase_node(new_ag_node)
            kwargs = dict(fused_node.kwargs)
            if "return_A" in kwargs:
                kwargs["return_A"] = False
                fused_node.kwargs = kwargs

    # Raise ancestors of non-A args that are topologically ordered between
    # ag_res_node and the matmul above fused_node.
    order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes_to_raise = sorted(
        OrderedSet(x for matmul in matmuls for x in matmul.arg_ancestor_nodes),
        key=lambda x: order[x],
    )
    for node in nodes_to_raise:
        if order[node] > order[fused_node]:
            fused_node.prepend(node)


def _find_producer_matmul(node: torch.fx.Node) -> Optional[_Matmul]:
    if node.target == aten.mm.default:
        return _Matmul.from_match(match=[node])
    elif node.target == aten._scaled_mm.default:
        return _ScaledMatmul.from_match(match=[node])
    elif node.target == aten.reshape.default:
        reshape_node_1 = node

        mm_node = reshape_node_1.args[0]
        assert isinstance(mm_node, torch.fx.Node)
        if mm_node.target not in (aten.mm.default, aten._scaled_mm.default):
            return None

        reshape_node_0 = mm_node.args[0]
        assert isinstance(reshape_node_0, torch.fx.Node)
        if reshape_node_0.target != aten.reshape.default:
            return None

        if mm_node.target == aten.mm.default:
            return _Matmul.from_match(match=[reshape_node_0, mm_node, reshape_node_1])
        elif mm_node.target == aten._scaled_mm.default:
            return _ScaledMatmul.from_match(
                match=[reshape_node_0, mm_node, reshape_node_1]
            )
    return None


def _insert_fused_matmul_reduce_scatter(
    graph: torch.fx.Graph,
    matmul: _Matmul,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.fx.Node:
    if type(matmul) == _Matmul:
        return graph.call_function(
            torch.ops.symm_mem.fused_matmul_reduce_scatter.default,
            args=(matmul.A_node, matmul.B_node, reduce_op, scatter_dim, group_name),
        )
    elif type(matmul) == _ScaledMatmul:
        return graph.call_function(
            torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter.default,
            args=(
                matmul.A_node,
                matmul.B_node,
                matmul.A_scale_node,
                matmul.B_scale_node,
                reduce_op,
                scatter_dim,
                group_name,
                matmul.bias_node,
                matmul.result_scale_node,
                matmul.out_dtype,
                matmul.use_fast_accum,
            ),
        )
    else:
        raise AssertionError(f"Unexpected matmul match type: {type(matmul)}")


def fuse_matmul_reduce_scatter(reduce_scatter: _ReduceScatterMatch) -> None:
    """
    Fused the pattern

        reduce_scatter_tensor(A @ B, scatter_dim, group_name)

    into

        torch.ops.symm_mem.fused_matmul_reduce_scatter(
            A, B, scatter_dim, group_name,
        )
    """
    if (
        not torch.distributed.is_available()
        or not torch.distributed.is_nccl_available()
    ):
        return

    from torch.distributed._symmetric_memory import (
        is_symm_mem_enabled_for_group,
        restride_A_for_fused_matmul_reduce_scatter,
    )

    input_node, _rs_node, rs_res_node, reduce_op, scatter_dim, group_name = (
        reduce_scatter.input_node,
        reduce_scatter.rs_node,
        reduce_scatter.res_node,
        reduce_scatter.reduce_op,
        reduce_scatter.scatter_dim,
        reduce_scatter.group_name,
    )

    if not is_symm_mem_enabled_for_group(group_name):
        return

    # Currently fused_matmul_reduce_scatter doesn't return the matmul result,
    # so we can't apply the fusion if the matmul result is used by multiple
    # users. This is not a fundamental limitation of the fused op and can be
    # addressed if needed.
    if len(input_node.users) != 1:
        return

    matmul = _find_producer_matmul(input_node)
    if matmul is None:
        return

    if rs_res_node in matmul.arg_ancestor_nodes:
        return

    graph = rs_res_node.graph
    with graph.inserting_before(rs_res_node):
        if "val" in matmul.A_node.meta:
            restrided = restride_A_for_fused_matmul_reduce_scatter(
                _get_tensor(matmul.A_node),
                scatter_dim,
            )
            matmul.A_node = graph.call_function(
                inductor_prims.force_stride_order,
                args=(matmul.A_node, restrided.stride()),
            )

        fused_node = _insert_fused_matmul_reduce_scatter(
            graph,
            matmul,
            reduce_op,
            scatter_dim,
            group_name,
        )
        reduce_scatter.replace_with(fused_node)
        reduce_scatter.erase()
        matmul.erase()

    order = {node: idx for idx, node in enumerate(graph.nodes)}
    nodes_to_raise = sorted(
        matmul.arg_ancestor_nodes,
        key=lambda x: order[x],
    )
    for node in nodes_to_raise:
        if order[node] > order[fused_node]:
            fused_node.prepend(node)


def _get_node_to_ancestors(
    graph: torch.fx.Graph,
) -> dict[torch.fx.Node, OrderedSet[torch.fx.Node]]:
    """
    Compute the ancestors for all nodes in a graph.
    """
    node_to_ancestors = defaultdict(OrderedSet[torch.fx.Node])  # type: ignore[var-annotated]
    for node in graph.nodes:
        node_to_ancestors[node] = OrderedSet(node.all_input_nodes)
        for dep in node.all_input_nodes:
            node_to_ancestors[node] |= node_to_ancestors[dep]

    return node_to_ancestors


def _get_collective_to_overlappable_nodes(
    graph: torch.fx.Graph,
) -> dict[torch.fx.Node, list[torch.fx.Node]]:
    """
    For each collective in the graph, find nodes that are neither ancestors nor
    descendants of the collective.
    """

    def is_collective(node) -> bool:
        # Only consider all-gather and reduce-scatter in the context of
        # micro-pipeline TP.
        return node.target in [
            torch.ops._c10d_functional.all_gather_into_tensor.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
        ]

    node_to_ancestors = _get_node_to_ancestors(graph)
    collective_to_overlappable_nodes = defaultdict(list)
    for node in graph.nodes:
        if not is_collective(node):
            continue
        for x in graph.nodes:
            if (
                node not in node_to_ancestors[x]
                and x not in node_to_ancestors[node]
                and x.op == "call_function"
            ):
                collective_to_overlappable_nodes[node].append(x)

    return collective_to_overlappable_nodes


def _get_unexposed_collectives(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    """
    Find all unexposed collectives in the graph.

    Because we don't have the runtime estimate, this function is a rough
    estimation using the following strong/hand-wavy assumptions:

    - Only a predefined set of "compute intensive" operation can hide a collective.
    - Any "compute intensive" operation can hide exactly one collective.
    """

    def _is_compute_intensive(node: torch.fx.Node) -> bool:
        return node.target in [torch.ops.aten.mm.default]

    collective_to_overlapping_candidates = defaultdict(list)
    available_nodes = OrderedSet[torch.fx.Node]()
    collective_to_overlappable_nodes = _get_collective_to_overlappable_nodes(graph)
    for collective, overlappable_nodes in collective_to_overlappable_nodes.items():
        candidates = [x for x in overlappable_nodes if _is_compute_intensive(x)]
        collective_to_overlapping_candidates[collective] = candidates
        available_nodes.update(candidates)

    unexposed_collectives = []
    for (
        collective,
        overlapping_candidates,
    ) in collective_to_overlapping_candidates.items():
        # Each collective consumes exactly one overlapping candidate
        for x in overlapping_candidates:
            if x in available_nodes:
                unexposed_collectives.append(collective)
                available_nodes.remove(x)
                break
    return unexposed_collectives


def micro_pipeline_tp_pass(graph: torch.fx.Graph):
    all_gathers = find_all_gather_patterns(graph)
    reduce_scatters = find_reduce_scatter_patterns(graph)

    # When a collective can be hidden through either simple overlapping or
    # micro-pipeline TP, we prefer simple overlapping to avoid the overhead
    # associated with decomposition. If reorder_for_compute_comm_overlap is
    # enabled, we identify collectives that can be hidden through simple
    # overlapping and exclude them from micro-pipeline TP candidates.
    if config.reorder_for_compute_comm_overlap:
        unexposed_collectives = _get_unexposed_collectives(graph)
        all_gathers = [x for x in all_gathers if x.ag_node not in unexposed_collectives]
        reduce_scatters = [
            x for x in reduce_scatters if x.rs_node not in unexposed_collectives
        ]

    for all_gather in all_gathers:
        fuse_all_gather_matmul(all_gather)

    for reduce_scatter in reduce_scatters:
        fuse_matmul_reduce_scatter(reduce_scatter)
