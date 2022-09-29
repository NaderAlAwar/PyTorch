# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional

from torch.onnx._internal.diagnostics.infra.sarif_om import (
    _edge_traversal,
    _message,
    _property_bag,
)


@dataclasses.dataclass
class GraphTraversal(object):
    """Represents a path through a graph."""

    description: Optional[_message.Message] = dataclasses.field(
        default=None, metadata={"schema_property_name": "description"}
    )
    edge_traversals: Optional[List[_edge_traversal.EdgeTraversal]] = dataclasses.field(
        default=None, metadata={"schema_property_name": "edgeTraversals"}
    )
    immutable_state: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "immutableState"}
    )
    initial_state: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "initialState"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    result_graph_index: int = dataclasses.field(
        default=-1, metadata={"schema_property_name": "resultGraphIndex"}
    )
    run_graph_index: int = dataclasses.field(
        default=-1, metadata={"schema_property_name": "runGraphIndex"}
    )


# flake8: noqa
