from enum import Enum
from typing import Any, Literal, Optional
from typing_extensions import TypeAlias

from torch._C import device, dtype, layout

# defined in torch/csrc/profiler/python/init.cpp

class RecordScope(Enum):
    FUNCTION = ...
    BACKWARD_FUNCTION = ...
    TORCHSCRIPT_FUNCTION = ...
    KERNEL_FUNCTION_DTYPE = ...
    CUSTOM_CLASS = ...
    BUILD_FEATURE = ...
    LITE_INTERPRETER = ...
    USER_SCOPE = ...
    STATIC_RUNTIME_OP = ...
    STATIC_RUNTIME_MODEL = ...

class ProfilerState(Enum):
    Disable = ...
    CPU = ...
    CUDA = ...
    NVTX = ...
    ITT = ...
    KINETO = ...
    KINETO_GPU_FALLBACK = ...
    KINETO_PRIVATEUSE1_FALLBACK = ...
    KINETO_PRIVATEUSE1 = ...

class ActiveProfilerType(Enum):
    NONE = ...
    LEGACY = ...
    KINETO = ...
    NVTX = ...
    ITT = ...

class ProfilerActivity(Enum):
    CPU = ...
    CUDA = ...
    XPU = ...
    MTIA = ...
    HPU = ...
    PrivateUse1 = ...

class _EventType(Enum):
    TorchOp = ...
    Backend = ...
    Allocation = ...
    OutOfMemory = ...
    PyCall = ...
    PyCCall = ...
    Kineto = ...

class _ExperimentalConfig:
    def __init__(
        self,
        profiler_metrics: list[str] = ...,
        profiler_measure_per_kernel: bool = ...,
        verbose: bool = ...,
        performance_events: list[str] = ...,
        enable_cuda_sync_events: bool = ...,
    ) -> None: ...

class ProfilerConfig:
    def __init__(
        self,
        state: ProfilerState,
        report_input_shapes: bool,
        profile_memory: bool,
        with_stack: bool,
        with_flops: bool,
        with_modules: bool,
        experimental_config: _ExperimentalConfig,
        trace_id: Optional[str] = None,
    ) -> None: ...

class _ProfilerEvent:
    start_tid: int
    start_time_ns: int
    children: list[_ProfilerEvent]

    # TODO(robieta): remove in favor of `self.typed`
    extra_fields: (
        _ExtraFields_TorchOp
        | _ExtraFields_Backend
        | _ExtraFields_Allocation
        | _ExtraFields_OutOfMemory
        | _ExtraFields_PyCall
        | _ExtraFields_PyCCall
        | _ExtraFields_Kineto
    )

    @property
    def typed(
        self,
    ) -> (
        tuple[Literal[_EventType.TorchOp], _ExtraFields_TorchOp]
        | tuple[Literal[_EventType.Backend], _ExtraFields_Backend]
        | tuple[Literal[_EventType.Allocation], _ExtraFields_Allocation]
        | tuple[Literal[_EventType.OutOfMemory], _ExtraFields_OutOfMemory]
        | tuple[Literal[_EventType.PyCall], _ExtraFields_PyCall]
        | tuple[Literal[_EventType.PyCCall], _ExtraFields_PyCCall]
        | tuple[Literal[_EventType.Kineto], _ExtraFields_Kineto]
    ): ...
    @property
    def name(self) -> str: ...
    @property
    def tag(self) -> _EventType: ...
    @property
    def id(self) -> int: ...
    @property
    def parent(self) -> _ProfilerEvent | None: ...
    @property
    def correlation_id(self) -> int: ...
    @property
    def end_time_ns(self) -> int: ...
    @property
    def duration_time_ns(self) -> int: ...

class _TensorMetadata:
    impl_ptr: int | None
    storage_data_ptr: int | None
    id: int | None

    @property
    def allocation_id(self) -> int | None: ...
    @property
    def layout(self) -> layout: ...
    @property
    def device(self) -> device: ...
    @property
    def dtype(self) -> dtype: ...
    @property
    def sizes(self) -> list[int]: ...
    @property
    def strides(self) -> list[int]: ...

Scalar: TypeAlias = int | float | bool | complex
Input: TypeAlias = _TensorMetadata | list[_TensorMetadata] | Scalar | None

class _ExtraFields_TorchOp:
    name: str
    sequence_number: int
    allow_tf32_cublas: bool

    @property
    def inputs(self) -> list[Input]: ...
    @property
    def scope(self) -> RecordScope: ...

class _ExtraFields_Backend: ...

class _ExtraFields_Allocation:
    ptr: int
    id: int | None
    alloc_size: int
    total_allocated: int
    total_reserved: int

    @property
    def allocation_id(self) -> int | None: ...
    @property
    def device(self) -> device: ...

class _ExtraFields_OutOfMemory: ...

class _PyFrameState:
    line_number: int
    function_name: str

    @property
    def file_name(self) -> str: ...

class _NNModuleInfo:
    @property
    def self_ptr(self) -> int: ...
    @property
    def cls_ptr(self) -> int: ...
    @property
    def cls_name(self) -> str: ...
    @property
    def parameters(
        self,
    ) -> list[tuple[str, _TensorMetadata, _TensorMetadata | None]]: ...

class _OptimizerInfo:
    @property
    def parameters(
        self,
    ) -> list[
        tuple[
            # Parameter
            _TensorMetadata,
            #
            # Gradient (if present during optimizer.step())
            _TensorMetadata | None,
            #
            # Optimizer state for Parameter as (name, tensor) pairs
            list[tuple[str, _TensorMetadata]],
        ]
    ]: ...

class _ExtraFields_PyCCall:
    @property
    def caller(self) -> _PyFrameState: ...

class _ExtraFields_PyCall:
    @property
    def callsite(self) -> _PyFrameState: ...
    @property
    def caller(self) -> _PyFrameState: ...
    @property
    def module(self) -> _NNModuleInfo | None: ...
    @property
    def optimizer(self) -> _OptimizerInfo | None: ...

class _ExtraFields_Kineto: ...

def _add_execution_trace_observer(output_file_path: str) -> bool: ...
def _remove_execution_trace_observer() -> None: ...
def _enable_execution_trace_observer() -> None: ...
def _disable_execution_trace_observer() -> None: ...
def _set_record_concrete_inputs_enabled_val(val: bool) -> None: ...
def _set_fwd_bwd_enabled_val(val: bool) -> None: ...
def _set_cuda_sync_enabled_val(val: bool) -> None: ...

class CapturedTraceback: ...

def gather_traceback(python: bool, script: bool, cpp: bool) -> CapturedTraceback: ...

# The Dict has name, filename, line
def symbolize_tracebacks(
    to_symbolize: list[CapturedTraceback],
) -> list[list[dict[str, str]]]: ...

class _RecordFunctionFast:
    def __init__(
        self,
        name: str,
        input_values: list | tuple | None = None,
        keyword_values: dict | None = None,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(self, *args: Any) -> None: ...
