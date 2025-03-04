# mypy: allow-untyped-defs
from __future__ import annotations


__all__ = [
    "VerificationInfo",
    "verify_onnx_program",
]

import dataclasses
import math
from typing import Any, TYPE_CHECKING

import torch
from torch.utils import _pytree
from torch.onnx._internal._lazy_import import onnxscript_ir as ir


if TYPE_CHECKING:
    from torch.onnx._internal.exporter import _onnx_program


@dataclasses.dataclass
class VerificationInfo:
    name: str
    max_abs_diff: float
    max_rel_diff: float
    abs_diff_hist: tuple[torch.Tensor, torch.Tensor]
    rel_diff_hist: tuple[torch.Tensor, torch.Tensor]
    expected_dtype: torch.dtype
    actual_dtype: torch.dtype
    # NOTE: We don't need to include shape because the expected shape is already known
    # and checked by the runtime


def _compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    # Move tensors to the same device
    expected = expected.detach().cpu()
    actual = actual.detach().cpu()
    if expected.numel() == 0 or actual.numel() == 0:
        return math.inf, math.inf, torch.tensor(math.inf), torch.tensor(math.inf)
    if expected.dtype == torch.bool:
        expected = expected.to(torch.float32)
        actual = actual.to(torch.float32)
    abs_diff = torch.abs(expected - actual)
    eps = 1e-7
    normalizer = torch.abs(expected) + eps
    rel_diff = abs_diff / normalizer

    max_absolute_difference = abs_diff.max().item()
    max_relative_difference = rel_diff.max().item()

    return max_absolute_difference, max_relative_difference, abs_diff, rel_diff


def verify_onnx_program(
    onnx_program: _onnx_program.ONNXProgram,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
) -> list[VerificationInfo]:
    exported_program = onnx_program.exported_program
    if exported_program is None:
        raise ValueError(
            "The ONNX program does not contain an exported_program. "
            "Please provide an exported_program to verify the ONNX program."
        )
    if args is None and kwargs is None:
        # User did not provide example inputs, use the default example inputs
        if exported_program.example_inputs is None:
            raise ValueError(
                "No example inputs provided and the exported_program does not contain example inputs. "
                "Please provide arguments to verify the ONNX program."
            )
        args, kwargs = exported_program.example_inputs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    torch_module = exported_program.module()
    torch_outputs, _ = _pytree.tree_flatten(torch_module(*args, **kwargs))
    onnx_outputs = onnx_program(*args, **kwargs)
    results = []
    for torch_output, onnx_output, output_val in zip(
        torch_outputs, onnx_outputs, onnx_program.model.graph.outputs
    ):
        name = output_val.name
        max_abs_diff, max_rel_diff, abs_diff, rel_diff = _compare_tensors(
            torch_output, onnx_output
        )
        abs_diff = abs_diff.flatten()
        rel_diff = rel_diff.flatten()
        bins = torch.tensor(
            [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10, 1000000],
            dtype=abs_diff.dtype,
        )
        abs_diff_hist = torch.histogram(abs_diff, bins=bins)
        rel_diff_hist = torch.histogram(rel_diff, bins=bins)
        results.append(
            VerificationInfo(
                name=str(name),
                max_abs_diff=max_abs_diff,
                max_rel_diff=max_rel_diff,
                abs_diff_hist=abs_diff_hist,
                rel_diff_hist=rel_diff_hist,
                expected_dtype=torch_output.dtype,
                actual_dtype=onnx_output.dtype,
            )
        )
    return results


def _create_value_mapping(graph: ir.Graph) -> dict[str, ir.Value]:
    """Return a dictionary mapping names to values in the graph.

    The mapping does not include values from subgraphs.

    Args:
        graph: The graph to extract the mapping from.

    Returns:
        A dictionary mapping names to values.
    """
    values = {}
    values.update(graph.initializers)
    # The names of the values can be None or "", which we need to exclude
    for input in graph.inputs:
        if not input.name:
            continue
        values[input.name] = input
    for node in graph:
        for value in node.outputs:
            if not value.name:
                continue
            values[value.name] = value
    return values


class VerificationInterpreter(torch.fx.Interpreter):
    def __init__(self, onnx_program: torch.onnx.ONNXProgram):
        if onnx_program.exported_program is None:
            raise ValueError(
                "The ONNX program does not contain an exported_program. "
                "Please provide an exported_program to verify the ONNX program."
            )
        super().__init__(onnx_program.exported_program.graph_module)
        self._onnx_program = onnx_program
        self._onnx_values = _create_value_mapping(onnx_program.model.graph)
        self.verification_info: list[VerificationInfo] = []
        self._args = []

    def run(
        self,
        *args,
        initial_env: dict[torch.fx.Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        self.verification_info = []
        self.args = args
        return super().run(
            *args,
            initial_env=initial_env,
            enable_io_processing=enable_io_processing,
        )

    def run_node(self, n: torch.fx.Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        result = super().run_node(n)
        node_name = n.name
        if node_name in self._onnx_values:
            # If the node name is in the ONNX values, we need to set the value
            # in the ONNX program
            (onnx_result,) = self._onnx_program.compute_values([node_name], self.args)
            max_absolute_difference, max_relative_difference, abs_diff, rel_diff = (
                _compare_tensors(
                    result,
                    onnx_result,
                )
            )
            self.verification_info.append(
                VerificationInfo(
                    name=node_name,
                    max_abs_diff=max_absolute_difference,
                    max_rel_diff=max_relative_difference,
                    abs_diff_hist=torch.histogram(abs_diff),
                    rel_diff_hist=torch.histogram(rel_diff),
                    expected_dtype=result.dtype,
                    actual_dtype=onnx_result.dtype,
                )
            )
        return result
