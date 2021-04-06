import torch
import torch.autograd.profiler as prof
import torch.utils.hooks as hooks
from torch.autograd import ProfilerActivity

from enum import Enum
from typing import Any, Callable, Iterable, Optional
from warnings import warn


class ProfilerAction(Enum):
    """
    Profiler actions that can be taken at the specified intervals
    """
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


def schedule(*, wait: int, warmup: int, active: int, repeat: int = 0) -> Callable:
    """
    Returns a callable that can be used as profiler ``schedule`` argument. The profiler will wait for ``wait`` steps, then
    do the warmup for the next ``warmup`` steps, then
    do the active recording for the next ``active`` steps and then
    repeat the cycle starting with the next step. The number of cycles is specified by the ``repeat`` parameter.
    When the parameter's value is zero, the cycles will continue until the profiling is finished.
    """
    def schedule_fn(step: int) -> ProfilerAction:
        assert step >= 0
        num_steps = wait + warmup + active
        if repeat > 0 and step / num_steps >= repeat:
            return ProfilerAction.NONE
        mod_step = step % num_steps
        if mod_step < wait:
            return ProfilerAction.NONE
        elif mod_step < wait + warmup:
            return ProfilerAction.WARMUP
        else:
            return ProfilerAction.RECORD if mod_step < num_steps - 1 \
                else ProfilerAction.RECORD_AND_SAVE
    assert wait >= 0 and warmup >= 0 and active > 0, \
        "Invalid profiler schedule arguments"
    if warmup == 0:
        warn("Profiler won't be using warmup, this can skew profiler results")
    return schedule_fn


def _default_schedule_fn(_: int) -> ProfilerAction:
    """
    Default profiler behavior - immediately starts recording the events,
    keeps doing it on every profiler step.
    """
    return ProfilerAction.RECORD


def tensorboard_trace_handler(dir_name: str, worker_name: Optional[str] = None):
    """
    Outputs tracing files to directory of ``dir_name``, then that directory can be
    directly delivered to tensorboard as logdir.
    ``worker_name`` should be unique for each worker in distributed scenario,
    it will be set to '[hostname]_[pid]' by default.
    """
    import os
    import socket
    import time

    def handler_fn(prof) -> None:
        nonlocal worker_name
        if not os.path.isdir(dir_name):
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                raise RuntimeError("Can't create directory: " + dir_name)
        if not worker_name:
            worker_name = "{}_{}".format(socket.gethostname(), str(os.getpid()))
        file_name = "{}.{}.pt.trace.json".format(worker_name, int(time.time() * 1000))
        prof.export_chrome_trace(os.path.join(dir_name, file_name))
    return handler_fn


module_level=0
module_record_functions = {}

def profiler_module_forward_pre_hook(module, input):
    global module_level
    # TODO: trace the top level module when the module_level is 0
    # DO we need add lock here?
    module_level = module_level + 1

    if (module_level == 1 or # top level module
        isinstance(module, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))):
        # TODO: trace the module children names when record_function add support of the extra arguments for children.
        record_func = prof.record_function(module.__class__.__name__)
        record_func.__enter__()
        module_record_functions[module] = record_func
    else:
        # TODO: start the record_function in case of the module view is enabled
        pass


def profiler_module_forward_hook(module, input, output):
    global module_level
    # DO we need add lock here?
    module_level = module_level - 1
    record_func = module_record_functions.pop(module, None)
    if record_func:
        record_func.__exit__(None, None, None)


class profile(object):
    """
    Profiler context manager.

    Args:

    - ``activities`` - list of activity groups (CPU, CUDA) to use in profiling, supported values:
      ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``;
      default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA;
    - ``schedule`` - callable that takes step (int) as a single parameter and returns
      ``ProfilerAction`` value that specifies the profiler action to perform at each step;
    - ``on_trace_ready`` - callable that is called at each step when ``schedule`` returns ``ProfilerAction.RECORD_AND_SAVE``
      during the profiling;
    - ``record_shapes`` - save information about operator's input shapes;
    - ``profile_memory`` - track tensor memory allocation/deallocation;
    - ``with_stack`` - record source information (file and line number) for the ops;
    - ``with_flops`` - use formula to estimate the FLOPS of specific operators (matrix multiplication and 2D convolution);
    - ``use_cuda`` - (deprecated, use ``activities``).

    .. note::
        Use ``torch.profiler.schedule`` to generate the callable schedule.
        Non-default schedules are useful when profiling long training jobs
        and allow the user to obtain multiple traces at the different iterations
        of the training process.
        The default schedule simply records all the events continuously for the
        duration of the context manager.

    .. note::
        Use ``torch.profiler.tensorboard_trace_handler`` to generate result files for TensorBoard:

        ``on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)``

        After profiling, result files can be found in the specified directory. Use the command:

        ``tensorboard --log_dir=dir_name``

        to see the results in TensorBoard.
        For more information, see `Pytorch Profiler <https://github.com/pytorch/kineto/tree/master/tb_plugin>`__

    .. note::
        Enabling shape and stack tracing results in additional overhead.

    Examples:

    .. code-block:: python

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA]
        ) as p:
            code_to_profile()
        print(p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    Using the profiler's ``schedule``, ``on_trace_ready`` and ``step`` functions:

    .. code-block:: python

        # Non-default profiler schedule allows user to turn profiler on and off
        # on different iterations of the training loop;
        # trace_handler is called every time a new trace becomes available
        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
            # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],

            # In this example with wait=1, warmup=1, active=2,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=trace_handler
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
            ) as p:
                for iter in range(N):
                    code_iteration_to_profile(iter)
                    # send a signal to the profiler that the next iteration has started
                    p.step()
    """
    def __init__(
            self,
            *,
            activities: Optional[Iterable[ProfilerActivity]] = None,
            schedule: Optional[Callable[[int], ProfilerAction]] = None,
            on_trace_ready: Optional[Callable[..., Any]] = None,
            record_shapes: bool = False,
            profile_memory: bool = False,
            with_stack: bool = False,
            with_flops: bool = False,
            # deprecated:
            use_cuda: Optional[bool] = None):
        if activities:
            self.activities = set(activities)
        else:
            self.activities = set([ProfilerActivity.CPU])
            if torch.cuda.is_available():
                self.activities.add(ProfilerActivity.CUDA)

        if use_cuda is not None:
            warn("use_cuda is deprecated, use activities argument instead")
            if use_cuda:
                self.activities.add(ProfilerActivity.CUDA)
            elif ProfilerActivity.CUDA in self.activities:
                self.activities.remove(ProfilerActivity.CUDA)

        assert len(self.activities) > 0, "No profiler activities specified"
        assert (ProfilerActivity.CUDA not in self.activities) or torch.cuda.is_available(), \
            "CUDA activity specified, but CUDA is not available"

        if schedule:
            self.schedule = schedule
            # add step markers into the trace and table view
            self.record_steps = True
        else:
            self.schedule = _default_schedule_fn
            self.record_steps = False
        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.step_num = 0
        self.current_action = self.schedule(self.step_num)
        self.profiler: Optional[prof.profile] = None
        self.step_rec_fn: Optional[prof.record_function] = None
        self.pre_forward_hook: Optional[hooks.RemovableHandle] = None
        self.forward_hook: Optional[hooks.RemovableHandle] = None

    def __enter__(self):
        self._enter_actions()
        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num))
            self.step_rec_fn.__enter__()

        self.pre_forward_hook = torch.nn.modules.module.register_module_forward_pre_hook(profiler_module_forward_pre_hook)
        self.forward_hook = torch.nn.modules.module.register_module_forward_hook(profiler_module_forward_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.forward_hook:
            self.forward_hook.remove()
            self.forward_hook = None
        if self.pre_forward_hook:
            self.pre_forward_hook.remove()
            self.pre_forward_hook = None

        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self._exit_actions()

    def step(self):
        """
        Signals the profiler that the next profiling step has started.
        """
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        prev_action = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)

        if self.current_action == ProfilerAction.NONE:
            if prev_action == ProfilerAction.NONE:
                pass
            elif prev_action == ProfilerAction.WARMUP:
                warn("Incorrect schedule: WARMUP followed by NONE")
                self._start_trace()
                self._stop_trace()
            elif prev_action == ProfilerAction.RECORD:
                warn("Incorrect schedule: RECORD followed by NONE")
                self._stop_trace()
            else:
                assert prev_action == ProfilerAction.RECORD_AND_SAVE
                self._stop_trace()
                if self.on_trace_ready:
                    self.on_trace_ready(self)
        elif self.current_action == ProfilerAction.WARMUP:
            if prev_action == ProfilerAction.NONE:
                self._start_warmup()
            elif prev_action == ProfilerAction.WARMUP:
                pass
            elif prev_action == ProfilerAction.RECORD:
                warn("Incorrect schedule: RECORD followed by WARMUP")
                self._stop_trace()
            else:
                assert prev_action == ProfilerAction.RECORD_AND_SAVE
                self._stop_trace()
                if self.on_trace_ready:
                    self.on_trace_ready(self)
                self._start_warmup()
        elif self.current_action in \
                [ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]:
            if prev_action == ProfilerAction.NONE:
                self._start_warmup()
                self._start_trace()
            elif prev_action == ProfilerAction.WARMUP:
                self._start_trace()
            elif prev_action == ProfilerAction.RECORD:
                pass
            else:
                assert prev_action == ProfilerAction.RECORD_AND_SAVE
                self._stop_trace()
                if self.on_trace_ready:
                    self.on_trace_ready(self)
                self._start_warmup()
                self._start_trace()

        if self.record_steps:
            self.step_rec_fn = prof.record_function("ProfilerStep#" + str(self.step_num))
            self.step_rec_fn.__enter__()

    def export_chrome_trace(self, path: str):
        """
        Exports the collected trace in Chrome JSON format.
        """
        assert self.profiler
        return self.profiler.export_chrome_trace(path)

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        """
        Save stack traces in a file in a format suitable for visualization.

        Args:

        - ``path`` - save stacks file to this location;
        - ``metric`` - metric to use: "self_cpu_time_total" or "self_cuda_time_total"

        .. note::
            Example of using FlameGraph tool:

            - git clone https://github.com/brendangregg/FlameGraph
            - cd FlameGraph
            - ./flamegraph.pl --title "CPU time" --countname "us." profiler.stacks > perf_viz.svg
        """
        assert self.profiler
        return self.profiler.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape: bool = False, group_by_stack_n: int = 0):
        """
        Averages events, grouping them by operator name and (optionally) input shapes and
        stack.
        Note: to use shape/stack functionality make sure to set record_shapes/with_stack
        when creating profiler context manager.
        """
        assert self.profiler
        return self.profiler.key_averages(group_by_input_shape, group_by_stack_n)

    def events(self):
        """
        Returns the list of unaggregated profiler events,
        to be used in the trace callback or after the profiling is finished
        """
        assert self.profiler
        return self.profiler.function_events

    def _enter_actions(self):
        if self.current_action == ProfilerAction.WARMUP:
            self._start_warmup()
        elif self.current_action in \
                [ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]:
            self._start_warmup()
            self._start_trace()

    def _exit_actions(self):
        if self.current_action == ProfilerAction.WARMUP:
            self._start_trace()
            self._stop_trace()
        elif self.current_action in \
                [ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]:
            self._stop_trace()
            if self.on_trace_ready:
                self.on_trace_ready(self)

    def _start_warmup(self):
        self.profiler = prof.profile(
            use_cuda=(ProfilerActivity.CUDA in self.activities),
            use_cpu=(ProfilerActivity.CPU in self.activities),
            record_shapes=self.record_shapes,
            with_flops=self.with_flops,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            use_kineto=True,
        )
        self.profiler._prepare_kineto_trace()

    def _start_trace(self):
        assert self.profiler is not None
        self.profiler._start_kineto_trace()

    def _stop_trace(self):
        assert self.profiler is not None
        self.profiler.__exit__(None, None, None)
