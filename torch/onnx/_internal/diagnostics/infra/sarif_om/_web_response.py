# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from torch.onnx._internal.diagnostics.infra.sarif_om import (
    _artifact_content,
    _property_bag,
)


@dataclasses.dataclass
class WebResponse(object):
    """Describes the response to an HTTP request."""

    body: Optional[_artifact_content.ArtifactContent] = dataclasses.field(
        default=None, metadata={"schema_property_name": "body"}
    )
    headers: Any = dataclasses.field(
        default=None, metadata={"schema_property_name": "headers"}
    )
    index: int = dataclasses.field(
        default=-1, metadata={"schema_property_name": "index"}
    )
    no_response_received: Optional[bool] = dataclasses.field(
        default=None, metadata={"schema_property_name": "noResponseReceived"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )
    protocol: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "protocol"}
    )
    reason_phrase: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "reasonPhrase"}
    )
    status_code: Optional[int] = dataclasses.field(
        default=None, metadata={"schema_property_name": "statusCode"}
    )
    version: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "version"}
    )


# flake8: noqa
