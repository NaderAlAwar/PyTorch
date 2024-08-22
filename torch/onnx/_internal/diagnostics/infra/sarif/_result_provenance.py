# DO NOT EDIT! This file was generated by jschema_to_python version 0.0.1.dev29,
# with extension for dataclasses and type annotation.

from __future__ import annotations

import dataclasses
from typing import List, Optional

from torch.onnx._internal.diagnostics.infra.sarif import (
    _physical_location,
    _property_bag,
)


@dataclasses.dataclass
class ResultProvenance(object):
    """Contains information about how and when a result was detected."""

    conversion_sources: Optional[List[_physical_location.PhysicalLocation]] = (
        dataclasses.field(
            default=None, metadata={"schema_property_name": "conversionSources"}
        )
    )
    first_detection_run_guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "firstDetectionRunGuid"}
    )
    first_detection_time_utc: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "firstDetectionTimeUtc"}
    )
    invocation_index: int = dataclasses.field(
        default=-1, metadata={"schema_property_name": "invocationIndex"}
    )
    last_detection_run_guid: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "lastDetectionRunGuid"}
    )
    last_detection_time_utc: Optional[str] = dataclasses.field(
        default=None, metadata={"schema_property_name": "lastDetectionTimeUtc"}
    )
    properties: Optional[_property_bag.PropertyBag] = dataclasses.field(
        default=None, metadata={"schema_property_name": "properties"}
    )


# flake8: noqa
