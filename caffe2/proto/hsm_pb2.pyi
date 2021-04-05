"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
"""
import builtins
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.message
import typing
import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor = ...

class NodeProto(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    CHILDREN_FIELD_NUMBER: int
    WORD_IDS_FIELD_NUMBER: int
    OFFSET_FIELD_NUMBER: int
    NAME_FIELD_NUMBER: int
    SCORES_FIELD_NUMBER: int
    word_ids: google.protobuf.internal.containers.RepeatedScalarFieldContainer[int] = ...
    offset: int = ...
    name: typing.Text = ...
    scores: google.protobuf.internal.containers.RepeatedScalarFieldContainer[float] = ...

    @property
    def children(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___NodeProto]: ...

    def __init__(self,
        *,
        children : typing.Optional[typing.Iterable[global___NodeProto]] = ...,
        word_ids : typing.Optional[typing.Iterable[int]] = ...,
        offset : typing.Optional[int] = ...,
        name : typing.Optional[typing.Text] = ...,
        scores : typing.Optional[typing.Iterable[float]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"name",b"name",u"offset",b"offset"]) -> bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"children",b"children",u"name",b"name",u"offset",b"offset",u"scores",b"scores",u"word_ids",b"word_ids"]) -> None: ...
global___NodeProto = NodeProto

class TreeProto(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    ROOT_NODE_FIELD_NUMBER: int

    @property
    def root_node(self) -> global___NodeProto: ...

    def __init__(self,
        *,
        root_node : typing.Optional[global___NodeProto] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"root_node",b"root_node"]) -> bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"root_node",b"root_node"]) -> None: ...
global___TreeProto = TreeProto

class HierarchyProto(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    SIZE_FIELD_NUMBER: int
    PATHS_FIELD_NUMBER: int
    size: int = ...

    @property
    def paths(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___PathProto]: ...

    def __init__(self,
        *,
        size : typing.Optional[int] = ...,
        paths : typing.Optional[typing.Iterable[global___PathProto]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"size",b"size"]) -> bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"paths",b"paths",u"size",b"size"]) -> None: ...
global___HierarchyProto = HierarchyProto

class PathProto(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    WORD_ID_FIELD_NUMBER: int
    PATH_NODES_FIELD_NUMBER: int
    word_id: int = ...

    @property
    def path_nodes(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___PathNodeProto]: ...

    def __init__(self,
        *,
        word_id : typing.Optional[int] = ...,
        path_nodes : typing.Optional[typing.Iterable[global___PathNodeProto]] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"word_id",b"word_id"]) -> bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"path_nodes",b"path_nodes",u"word_id",b"word_id"]) -> None: ...
global___PathProto = PathProto

class PathNodeProto(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor = ...
    INDEX_FIELD_NUMBER: int
    LENGTH_FIELD_NUMBER: int
    TARGET_FIELD_NUMBER: int
    index: int = ...
    length: int = ...
    target: int = ...

    def __init__(self,
        *,
        index : typing.Optional[int] = ...,
        length : typing.Optional[int] = ...,
        target : typing.Optional[int] = ...,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions.Literal[u"index",b"index",u"length",b"length",u"target",b"target"]) -> bool: ...
    def ClearField(self, field_name: typing_extensions.Literal[u"index",b"index",u"length",b"length",u"target",b"target"]) -> None: ...
global___PathNodeProto = PathNodeProto
