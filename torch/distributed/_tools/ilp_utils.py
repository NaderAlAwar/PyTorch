import copy
import weakref
from typing import cast, Dict, List, OrderedDict, Set, Tuple, TypedDict

import numpy as np

import torch
from torch.distributed._tools.mem_tracker import (
    _MemRefType,
    _ModMemStats,
    _ModState,
    MemTracker,
)
from torch.distributed._tools.runtime_estimator import RuntimeEstimator
from torch.distributed._tools.sac_estimator import (
    SACEstimator,
    SACGreedyOrderMeta,
    SACStats,
    SACTradeOffStats,
)


def get_module_name_dict(root_module: torch.nn.Module) -> weakref.WeakKeyDictionary:
    """
    Create a weak key dictionary of modules and their fully qualified names.

    Args:
        root_module (torch.nn.Module): Root module to start traversal.

    Returns:
        weakref.WeakKeyDictionary: Dictionary of modules and their names.
    """
    module_dict: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def _get_mod_name(mod: torch.nn.Module, fqn: str = "") -> str:
        if mod in module_dict:
            return module_dict[mod]
        mod_name = fqn or type(mod).__name__
        module_dict[mod] = mod_name
        for name, submod in mod.named_children():
            _get_mod_name(submod, f"{mod_name}.{name}")
        return mod_name

    _get_mod_name(root_module)
    return module_dict


class ModOrder(TypedDict):
    fw_pre_order: List[str]
    bw_pre_order: List[str]
    fw_post_order: List[str]
    bw_post_order: List[str]


class ModRuntime(TypedDict):
    fw: float
    bw: float


class ModStats(TypedDict):
    fqn: str
    # per-module params
    param_per_module: int
    # per-module grads
    grad_per_module: int
    # total accumulated gradients up to and including this module
    grad_total: int
    # per module fw activation size (excluding input and output)
    act_fw_per_module: int
    # per module bw activation size during peak_bw
    act_bw_per_module: int
    # per module activation grad size during peak_bw
    act_grad_per_module: int
    # total activation size up to but excluding the current module
    # includes input of the current module (i.e., output of previous module)
    act_total: int
    # Inputs to the module
    input_per_module: int
    # Outputs of the module
    output_per_module: int
    # Total fw run-time of the module
    fw_runtime_per_module: float
    # Total bw run-time of the module
    bw_runtime_per_module: float
    # Is this module a leaf module
    is_leaf: bool
    # Total ac run-time of the module
    sac_runtime: float
    # Total ac_memory for the module
    sac_memory: int
    # The mandatory saved activation memory for a module
    saved_memory: int
    # Number of piecewise-linear functions used for approximating ac tradeoff curve
    n_segments: int
    # Slopes of the of piecewise-linear functions
    slopes: List[float]
    # Intercepts of the of piecewise-linear functions
    intercepts: List[float]
    # X breakpoints of the of piecewise-linear functions
    breakpoints: List[float]
    # Original trade-off curves
    tradeoff_curve: OrderedDict[float, float]


class ModuleInfo(TypedDict):
    mod_order: ModOrder
    mod_stats: List[ModStats]
    opt_memory: int


def aggregate_stats(
    model: torch.nn.Module,
    mem_tracker: MemTracker,
    runtime_estimator: RuntimeEstimator,
    sac_estimator: SACEstimator,
    dev: torch.device,
) -> ModuleInfo:
    """
    Collect modulewise stats for a given model, including memory, runtime, and AC tradeoff stats.

    Args:
        model: nn.Module object
        runtime_estimator: RuntimeEstimator object with runtime stats
        mem_tracker: MemTracker object with memory stats
        sac_estimator: SACEstimator object with AC tradeoff stats
        dev: device the model was run on (used to extract memory stats from MemTracker)

    Returns:
        ModuleInfo: A dictionary with module order and module stats.
    """

    # Memory stats
    mod_mem_stats: Dict[torch.nn.Module, _ModMemStats] = dict(
        copy.deepcopy(mem_tracker.memory_tracking)
    )
    opt_memory = mem_tracker.get_tracker_snapshot("peak")[dev][_MemRefType.OPT]

    # Runtime stats
    mod_runtime_stats: Dict[str, ModRuntime] = {
        fqn: {"fw": v["fw"], "bw": v["bw"]}
        for fqn, v in runtime_estimator.mod_runtimes.items()
    }

    # Module order
    mod_order: ModOrder = {
        "fw_pre_order": list(runtime_estimator.mod_fw_pre_order),
        "bw_pre_order": list(runtime_estimator.mod_bw_pre_order),
        "fw_post_order": list(runtime_estimator.mod_fw_post_order),
        "bw_post_order": list(runtime_estimator.mod_bw_post_order),
    }

    # Selective Activation Checkpointing stats
    sac_estimator.pwlf_sac_tradeoff_curve()
    mod_sac_tradeoff_stats: Dict[str, SACTradeOffStats] = copy.deepcopy(
        sac_estimator.sac_mod_tradeoff_stats
    )
    mod_sac_greedy_stats: Dict[str, SACGreedyOrderMeta] = copy.deepcopy(
        sac_estimator.sac_mod_greedy_order_meta
    )

    mod_sac_stats: Dict[str, SACStats] = copy.deepcopy(sac_estimator.sac_mod_stats)

    module_info: ModuleInfo = {
        "mod_order": mod_order,
        "mod_stats": [],
        "opt_memory": opt_memory,
    }

    for mod in model.modules():
        if mod_mem_stat := mod_mem_stats.get(mod, None):
            if tradeoff_stats := mod_sac_tradeoff_stats.get(mod_mem_stat.mod_fqn, None):
                sac_runtime = tradeoff_stats.sac_runtime
                sac_memory = tradeoff_stats.sac_memory
                n_segments = tradeoff_stats.n_segments
                slopes = tradeoff_stats.slopes
                intercepts = tradeoff_stats.intercepts
                breakpoints = tradeoff_stats.fit_breaks
                tradeoff_curve = tradeoff_stats.tradeoff_curve
                is_leaf = False

                sac_stats_memory = mod_sac_stats[mod_mem_stat.mod_fqn].memory
                greedy_meta = mod_sac_greedy_stats[mod_mem_stat.mod_fqn]
                stored_ops, inplace_op_groups, random_inplace_ops = (
                    greedy_meta.stored_ops,
                    greedy_meta.inplace_op_groups,
                    greedy_meta.random_inplace_ops,
                )
                stored_indices: Set[int] = set()
                for s_idx in stored_ops:
                    stored_indices.add(s_idx)
                    if s_idx in inplace_op_groups:
                        stored_indices.update(inplace_op_groups[s_idx])
                    if s_idx in random_inplace_ops:
                        stored_indices.update(random_inplace_ops)
                saved_memory = sum(
                    sac_stats_memory[op_idx] for op_idx in stored_indices
                )
            else:
                sac_runtime = sac_memory = n_segments = saved_memory = 0
                slopes = intercepts = breakpoints = []
                tradeoff_curve: OrderedDict[float, float] = OrderedDict()  # type: ignore[no-redef]
                is_leaf = True
            mod_stat: ModStats = {
                "fqn": mod_mem_stat.mod_fqn,
                "param_per_module": mod_mem_stat.parameter_mem,
                "grad_per_module": mod_mem_stat.parameter_mem,
                "grad_total": mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                    _MemRefType.GRAD
                ],
                "act_fw_per_module": max(
                    0,
                    mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][_MemRefType.ACT]
                    - mod_mem_stat.snapshots[_ModState.PRE_FW][-1][dev][_MemRefType.ACT]
                    - mod_mem_stat.output_mem,
                ),
                "act_bw_per_module": max(
                    0,
                    mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.ACT],
                ),
                "act_grad_per_module": (
                    mod_mem_stat.snapshots[_ModState.PEAK_BW][-1][dev][_MemRefType.TEMP]
                    - mod_mem_stat.snapshots[_ModState.PRE_BW][-1][dev][
                        _MemRefType.TEMP
                    ]
                ),
                "act_total": mod_mem_stat.snapshots[_ModState.POST_FW][-1][dev][
                    _MemRefType.ACT
                ],
                "input_per_module": mod_mem_stat.input_mem,
                "output_per_module": mod_mem_stat.output_mem,
                "fw_runtime_per_module": mod_runtime_stats[mod_mem_stat.mod_fqn]["fw"],
                "bw_runtime_per_module": mod_runtime_stats[mod_mem_stat.mod_fqn]["bw"],
                "is_leaf": is_leaf,
                "sac_runtime": sac_runtime,
                "sac_memory": sac_memory,
                "saved_memory": saved_memory,
                "n_segments": n_segments,
                "slopes": slopes,
                "intercepts": intercepts,
                "breakpoints": breakpoints,
                "tradeoff_curve": tradeoff_curve,
            }
            module_info["mod_stats"].append(mod_stat)

    return module_info


class Node(ModStats):
    index: int  # index according to forward pre-order
    pos_fw_post_order: int  # index according to forward post-order


class Graph:
    def __init__(self, n: int) -> None:
        self.nodes: List[Node] = []
        self.name2node: Dict[str, Node] = {}
        self.ad_matrix = np.zeros((n, n))
        self.fw_post_order: List[str] = []
        self.opt_memory = 0

    def add_node(self, node: Node) -> None:
        self.nodes.append(node)
        self.name2node[node["fqn"]] = node


def parse_module_info(module_info: ModuleInfo) -> Graph:
    """
    Parse module info and create a graph (tree) of modules. The graph will be
    used by MILP solver to find optimal SAC and/or FSDP configurations.
    """
    mod_stats = module_info["mod_stats"]
    opt_memory = module_info["opt_memory"]
    fw_pre_order = module_info["mod_order"]["fw_pre_order"]
    # assertion and number of nodes
    assert len(mod_stats) == len(fw_pre_order)
    n_nodes = len(mod_stats)

    # create graph
    g = Graph(n_nodes)
    g.fw_post_order = module_info["mod_order"]["fw_post_order"]
    g.opt_memory = opt_memory

    # sort the modules by pre-order and add them to the graph
    module_info["mod_stats"] = sorted(
        mod_stats, key=lambda x: fw_pre_order.index(x["fqn"])
    )
    for i, one_mod_stats in enumerate(mod_stats):
        node: Node = cast(Node, one_mod_stats)
        node["index"] = i
        node["pos_fw_post_order"] = g.fw_post_order.index(node["fqn"])
        g.add_node(node)

    # set up ancestor-descendant matrix
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if is_self_or_submodule(g.nodes[j]["fqn"], g.nodes[i]["fqn"]):
                g.ad_matrix[i][j] = 1
            else:
                break

    return g


def is_self_or_submodule(name_descendant: str, name_ancestor: str) -> bool:
    """
    check if name_descendant is a submodule of name_ancestor, or if they are the same
    """
    return name_descendant == name_ancestor or name_ancestor + "." in name_descendant


def is_submodule(name_descendant: str, name_ancestor: str) -> bool:
    """
    if name_descendant is a submodule of name_ancestor, but not the same
    """
    return name_ancestor + "." in name_descendant


def display_bytes(b: int, unit: str = "MiB") -> str:
    """
    return a string that represent the number of bytes in a desired unit
    """
    if unit == "KiB":
        return f"{b/2**10:.2f} KiB"
    if unit == "MiB":
        return f"{b/2**20:.2f} MiB"
    if unit == "GiB":
        return f"{b/2**30:.2f} GiB"
    return f"{b:.2f} bytes"


def get_peak_memory_runtime_baseline(graph: Graph) -> Tuple[int, float]:
    """
    Get the baseline peak memory and runtime.
    Baseline here means there is no FSDP or AC.
    Memory includes the parameters, gradients, activations, and activation gradients.
    Memory does not include e.g., optimizer states, embedding tables, etc.

    Returns:
        int: peak memory in bytes
        float: compute time in ms
    """
    P_1 = graph.nodes[0]["param_per_module"]
    num_nodes = len(graph.nodes)
    peak_mem = 0
    for i in range(num_nodes):
        TG_i = graph.nodes[i]["grad_total"]
        AG_i = graph.nodes[i]["act_grad_per_module"]
        TA_i = graph.nodes[i]["act_total"]
        peak_mem = max(peak_mem, P_1 + TG_i + AG_i + TA_i)
    compute_time = (
        graph.nodes[0]["fw_runtime_per_module"]
        + graph.nodes[0]["bw_runtime_per_module"]
    )
    return (peak_mem, compute_time)
