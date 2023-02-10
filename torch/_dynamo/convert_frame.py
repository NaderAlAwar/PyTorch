import functools
import itertools
import logging
import os
import types
import weakref
from typing import Dict, Optional, Set

import torch
from torch.fx.graph_module import _forward_from_src as original_forward_from_src

from . import config, exc
from .allowed_functions import is_allowed
from .backends.registry import CompilerFn
from .bytecode_analysis import remove_dead_code, remove_pointless_jumps
from .bytecode_transformation import is_generator, transform_code_object
from .eval_frame import always_optimize_code_objects, skip_code, TorchPatcher
from .exc import (
    augment_exc_message,
    BackendCompilerFailed,
    format_error_msg,
    InternalTorchDynamoError,
    TorchRuntimeError,
    unimplemented,
    Unsupported,
)
from .guards import CheckFunctionManager, GuardedCode
from .hooks import Hooks
from .output_graph import OutputGraph
from .replay_record import ExecutionRecord
from .symbolic_convert import InstructionTranslator
from .utils import (
    CleanupManager,
    counters,
    dynamo_timed,
    format_bytecode,
    gen_record_file_name,
    guard_failures,
    increment_frame,
    init_logging,
    is_namedtuple,
    istype,
    orig_code_map,
    troubleshooting_url,
    write_record_to_file,
)

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self):
        self.seen = []
        self.seen_ids = set()

    def add(self, strong_obj):
        idx = id(strong_obj)
        if idx not in self.seen_ids:
            obj = weakref.ref(strong_obj, lambda _: self.seen_ids.remove(idx))
            self.seen.append(obj)
            self.seen_ids.add(idx)

    def __contains__(self, item):
        return id(item) in self.seen_ids

    def clear(self):
        self.seen.clear()
        self.seen_ids.clear()


input_codes = Tracker()
output_codes = Tracker()


initial_grad_state = None


@functools.wraps(original_forward_from_src)
def fx_forward_from_src_skip_result(*args, **kwargs):
    # we monkey patch FX to prevent infinite loop of trying to convert
    # our generated code
    result: types.FunctionType = original_forward_from_src(*args, **kwargs)
    skip_code(result.__code__)
    return result


def wrap_convert_context(fn):
    """
    Context manager to:
        1) Save/restore torch random state
        2) Save/restore torch.is_grad_enabled() state
        3) Monkey patch torch.fx.graph_module._forward_from_src
    """

    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        prior_grad_mode = torch.is_grad_enabled()
        rng_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            cuda_rng_state = torch.cuda.get_rng_state()
        prior_fwd_from_src = torch.fx.graph_module._forward_from_src
        torch.fx.graph_module._forward_from_src = fx_forward_from_src_skip_result
        try:
            return fn(*args, **kwargs)
        finally:
            torch._C._set_grad_enabled(prior_grad_mode)
            torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
            torch.fx.graph_module._forward_from_src = prior_fwd_from_src

    _fn._torchdynamo_orig_callable = fn  # type: ignore[attr-defined]
    return _fn


@TorchPatcher.suppress_torch_distributed_warnings
def has_tensor_in_frame(frame):
    """Check if the frame has torch.* related bits"""
    # Check if the function was decorated using torch._dynamo.optimize
    if frame.f_code in always_optimize_code_objects:
        return True

    # Check if there is global import of torch.*
    for co_name in frame.f_code.co_names:
        if co_name in frame.f_globals:
            if is_allowed(frame.f_globals[co_name]):
                return True

    seen_ids: Dict[int, bool] = dict()

    def has_tensor(obj):
        """Recursively check if the obj has a tensor"""
        obj_id = id(obj)
        if obj_id in seen_ids:
            return seen_ids[obj_id]
        seen_ids[obj_id] = False

        if isinstance(obj, (torch.Tensor, torch.nn.Module)):
            seen_ids[obj_id] = True
            return seen_ids[obj_id]
        elif istype(obj, (list, tuple)):
            seen_ids[obj_id] = any([has_tensor(v) for v in obj])
            return seen_ids[obj_id]
        elif istype(obj, dict):
            # Some packages like pytest can be updated during runtime. So, make a
            # copy of values to avoid issues like "RuntimeError: dictionary
            # changed size during iteration"
            values = list(obj.values())
            seen_ids[obj_id] = any([has_tensor(v) for v in values])
            return seen_ids[obj_id]
        elif istype(obj, (str, int, float, type(None), bool)):
            seen_ids[obj_id] = False
            return seen_ids[obj_id]
        elif is_namedtuple(obj):
            seen_ids[obj_id] = any([has_tensor(getattr(obj, v)) for v in obj._fields])
            return seen_ids[obj_id]
        else:
            # if config.debug:
            #     print(
            #         f"Assuming that object of type {type(obj)} does not have a tensor"
            #     )
            return False

    # Check if the passed arguments are of type Tensor
    for value in frame.f_locals.values():
        if has_tensor(value):
            return True

    log.debug(
        f"skipping because no torch.* {frame.f_code.co_name} \
            {frame.f_code.co_filename} {frame.f_code.co_firstlineno}"
    )

    return False


def exception_handler(e, code, frame=None):
    record_filename = None
    if hasattr(e, "exec_record"):
        record_filename = gen_record_file_name(e, code)
        write_record_to_file(record_filename, e.exec_record)
        e.record_filename = record_filename

    augment_exc_message(e)
    # Only log the exception if we are going to suppress it
    # if aren't suppressing it, a higher level except block will handle it
    if config.suppress_errors:
        log.error(format_error_msg(e, code, record_filename, frame))


def convert_frame_assert(
    compiler_fn: CompilerFn,
    one_graph: bool = True,
    export: bool = False,
):
    """Fully convert a frame into an FX graph"""
    init_logging()

    def _convert_frame_assert(frame: types.FrameType, cache_size: int, hooks: Hooks):
        increment_frame()
        code = frame.f_code
        input_codes.add(code)
        if code in output_codes:
            return None
        if (
            os.environ.get("TORCHDYNAMO_DEBUG_FUNCTION")
            and os.environ.get("TORCHDYNAMO_DEBUG_FUNCTION") != code.co_name
        ):
            return None
        if code.co_name == "<genexpr>" and code.co_filename.endswith(
            ("transformers/file_utils.py", "transformers/utils/generic.py")
        ):
            # not needed, but cleans up torchbench error stats
            return None
        if code.co_name == "__setattr__":
            # setattr could be tricky to handle generally,
            # but also not likely useful to compile- skip the whole frame
            return None

        # Check if the frame is generated by an exec builtin call
        # TODO - Running exec generated frame seems propagates f_globals to the
        # next frames.
        if code.co_name == "<module>" and code.co_filename == "<string>":
            return None

        if (
            code.co_name == "<lambda>"
            and code.co_filename == "<string>"
            and not bool(frame.f_builtins)
        ):
            # namedtuple subclass constructor. Empty builtins cause issue with
            # len keyword in LIST_LEN guard.
            return None

        if is_generator(code):
            unimplemented("generator")
        if cache_size >= config.cache_size_limit:

            def format_func_info(code):
                return f"'{code.co_name}' ({code.co_filename}:{code.co_firstlineno})"

            def format_guard_failures(code):
                # For the common case, it's sufficient to see just the most recent failure.
                # We could add a verbose mode if needed
                return f"{str(guard_failures[code][-1])}"

            assert code in guard_failures, "TODO(whc) any other recompile reasons?"
            log.warning(
                f"torch._dynamo hit config.cache_size_limit ({config.cache_size_limit})\n"
                + f"   function: {format_func_info(code)}\n"
                + f"   reasons:  {format_guard_failures(code)}\n"
                + f"to diagnose recompilation issues, see {troubleshooting_url}."
            )
            unimplemented("cache_size_limit reached")

        if not has_tensor_in_frame(frame):
            return None

        global initial_grad_state
        initial_grad_state = torch.is_grad_enabled()

        return _compile(
            frame.f_code,
            frame.f_globals,
            frame.f_locals,
            frame.f_builtins,
            compiler_fn,
            one_graph,
            export,
            hooks,
            frame,
        )

    _convert_frame_assert._torchdynamo_orig_callable = compiler_fn  # type: ignore[attr-defined]
    return wrap_convert_context(_convert_frame_assert)


@dynamo_timed(phase_name="entire_frame_compile")
def _compile(
    code: types.CodeType,
    globals: Dict[str, object],
    locals: Dict[str, object],
    builtins: Dict[str, object],
    compiler_fn: CompilerFn,
    one_graph: bool,
    export: bool,
    hooks: Hooks,
    frame: Optional[types.FrameType] = None,
) -> Optional[GuardedCode]:

    output: Optional[OutputGraph] = None
    # This is shared across restarts
    mutated_closure_cell_contents: Set[str] = set()

    # from .utils import print_once;  print_once(code.co_filename)

    def transform(instructions, code_options):
        nonlocal output
        tracer = InstructionTranslator(
            instructions,
            code,
            locals,
            globals,
            builtins,
            code_options,
            compiler_fn,
            one_graph,
            export,
            mutated_closure_cell_contents,
        )
        tracer.run()
        output = tracer.output
        assert output is not None
        assert output.output_instructions
        instructions[:] = output.output_instructions
        code_options.update(output.code_options)

        if config.dead_code_elimination:
            instructions[:] = remove_pointless_jumps(remove_dead_code(instructions))

    try:
        for attempt in itertools.count():
            try:
                out_code = transform_code_object(code, transform)
                orig_code_map[out_code] = code
                break
            except exc.RestartAnalysis:
                log.debug("Restarting analysis ...")
                if attempt > 100:
                    unimplemented("100+ RestartAnalysis() calls")
            except exc.SkipFrame as e:
                log.debug(
                    f"Skipping frame {e} {code.co_name} \
                    {code.co_filename} {code.co_firstlineno}"
                )
                if one_graph:
                    log.debug("No graph captured with one_graph=True")
                return None
        output_codes.add(out_code)

        if config.output_code:
            log.info(
                format_bytecode(
                    "ORIGINAL BYTECODE",
                    code.co_name,
                    code.co_filename,
                    code.co_firstlineno,
                    code,
                ),
            )
            log.info(
                format_bytecode(
                    "MODIFIED BYTECODE",
                    code.co_name,
                    code.co_filename,
                    code.co_firstlineno,
                    out_code,
                ),
            )

        assert output is not None
        assert output.guards is not None
        CleanupManager.instance[out_code] = output.cleanups
        check_fn = CheckFunctionManager(
            output,
            locals,
            globals,
            hooks.guard_fail_fn if hooks else None,
        )

        guarded_code = GuardedCode(out_code, check_fn.check_fn)

        if config.output_code:
            guard_str = "GUARDS:\n"
            guard_str += "\n".join(
                [f" - {str(guard)}" for guard in sorted(output.guards)]
            )
            log.info(guard_str)

        if hooks.guard_export_fn is not None:
            hooks.guard_export_fn(output.guards)

        return guarded_code
    except (
        Unsupported,
        TorchRuntimeError,
        BackendCompilerFailed,
        AssertionError,
    ) as e:
        exception_handler(e, code, frame)
        raise
    except Exception as e:
        exception_handler(e, code, frame)
        raise InternalTorchDynamoError() from e


def convert_frame(compiler_fn: CompilerFn, hooks: Hooks):
    """Try to convert a frame into an FX graph, if error leave frame unmodified"""
    inner_convert = convert_frame_assert(compiler_fn, one_graph=False)

    def _convert_frame(frame: types.FrameType, cache_size: int, hooks: Hooks):
        counters["frames"]["total"] += 1
        try:
            result = inner_convert(frame, cache_size, hooks)
            counters["frames"]["ok"] += 1
            return result
        except (NotImplementedError, Unsupported):
            log.info("converting frame raised unsupported, leaving it unconverted")
        except Exception:
            if not config.suppress_errors:
                raise
            log.info("converting frame raised error, suppressing error")
        return None

    _convert_frame._torchdynamo_orig_callable = compiler_fn  # type: ignore[attr-defined]
    return _convert_frame


# TODO mlazos: add support for same args, or record them
def replay(filename):
    from .backends.debugging import eager

    original_replay_val = config.replay_record_enabled
    config.replay_record_enabled = False
    init_logging()
    with open(filename, "rb") as in_file:
        record = ExecutionRecord.load(in_file)
    record.globals = {
        k: v for k, v in itertools.chain(record.globals.items(), globals().items())
    }

    try:
        _compile(
            record.code,
            record.globals,
            record.locals,
            record.builtins,
            compiler_fn=eager,
            one_graph=False,
            export=False,
            hooks=Hooks(),
            frame=None,
        )
    except Exception:
        pass
    finally:
        config.replay_record_enabled = original_replay_val
