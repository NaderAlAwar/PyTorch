# Copyright (c) Meta Platforms, Inc. and affiliates

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.distributed as dist
from torch.profiler import record_function

from .microbatch import merge_chunks, split_args_kwargs_into_chunks
from .PipelineStage import _PipelineStageBase

if TYPE_CHECKING:
    from ._IR import Pipe


__all__ = [
    "PipelineScheduleSingle",
    "PipelineScheduleMulti",
    "Schedule1F1B",
    "ScheduleGPipe",
    "ScheduleInterleaved1F1B",
    "ScheduleLoopedBFS",
]

logger = logging.getLogger(__name__)


class _PipelineSchedule(ABC):
    def __init__(
        self,
        n_microbatches: int,
        loss_fn: Optional[Callable[..., torch.Tensor]] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # From arguments
        self._n_microbatches = n_microbatches
        self._loss_fn = loss_fn
        self._output_merge_spec = output_merge_spec
        # Derived
        self._has_backward = self._loss_fn is not None
        # To be filled by subclasses
        self._pipe_info: Optional[Pipe.PipeInfo] = None

        # Holds the losses for each microbatch.
        self._internal_losses: List[torch.Tensor] = []
        logger.info(f"Using {self.__class__.__name__}")  # noqa: G004

    def _maybe_compute_loss(self, stage, output, target_mbs, mb_index):
        if stage.is_last and self._has_backward:
            loss = self._compute_loss(output, target_mbs[mb_index])  # type: ignore[index]
            self._internal_losses.append(loss)

    def _maybe_get_loss(self, stage, mb_index):
        valid_index = 0 <= mb_index < len(self._internal_losses)
        if stage.is_last and self._has_backward and valid_index:
            return self._internal_losses[mb_index]
        elif len(self._internal_losses) != 0 and not valid_index:
            raise RuntimeError(
                f"Loss for microbatch {mb_index} is not available. "
                f"Available losses for microbatches: {self._internal_losses}"
            )
        else:
            return None

    def _update_losses(self, stages, losses):
        """
        Update the losses to those in the internal state
        """
        # if stages not a list turn into a list
        if not isinstance(stages, list):
            stages = [stages]
        contains_last_stage = any(stage.is_last for stage in stages)

        # Return losses if there is a container passed in
        if contains_last_stage and losses is not None:
            if len(self._internal_losses) != self._n_microbatches:
                raise RuntimeError(
                    f"Expecting {self._n_microbatches} losses but got {len(self._internal_losses)}"
                )

            # Clean external container first
            losses.clear()
            # Copy internal losses to external container
            losses.extend(self._internal_losses)

        self._internal_losses.clear()

    @abstractmethod
    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the schedule
        implementation.

        Args:
            microbatches: list of microbatch args.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        """
        Run one iteration of the pipeline schedule with *whole-batch* input.
        Will chunk the input into microbatches automatically, and go through the
        microbatches according to the schedule implementation.

        args: positional arguments to the model (as in non-pipeline case).
        kwargs: keyword arguments to the model (as in non-pipeline case).
        target: target for the loss function.
        losses: a list to store the losses for each microbatch.
        """
        raise NotImplementedError

    def _check_inputs(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Pre-process/check inputs
        """

        def check_type_and_len(mbs, name: str):
            if not isinstance(mbs, list):
                raise TypeError(f"{name} must be a list but got a {type(mbs)}")
            if len(mbs) != self._n_microbatches:
                raise ValueError(
                    f"Expecting {self._n_microbatches} {name} but got {len(mbs)}"
                )

        if arg_mbs is not None:
            check_type_and_len(arg_mbs, "arg_mbs")
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            check_type_and_len(kwarg_mbs, "kwarg_mbs")
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        if target_mbs is not None:
            check_type_and_len(target_mbs, "target_mbs")

        if losses is not None:
            if not isinstance(losses, list):
                raise TypeError(f"losses must be a list but got a {type(losses)}")

        return arg_mbs, kwarg_mbs

    def _compute_loss(self, output, target):
        return self._loss_fn(output, target)  # type: ignore[misc]

    def _split_inputs(
        self,
        args: Tuple[Any, ...],
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Splits a full-batch input into chunks (i.e. microbatches) and returns
        the chunks
        """
        if self._pipe_info is not None:
            # Use spec from `pipe_info`
            args_chunk_spec = self._pipe_info.args_chunk_spec
            kwargs_chunk_spec = self._pipe_info.kwargs_chunk_spec
        else:
            # Use default spec from `microbatch.py` (i.e. chunk dim 0 for each arg/kwarg)
            args_chunk_spec = None
            kwargs_chunk_spec = None

        if args or kwargs:
            args_split, kwargs_split = split_args_kwargs_into_chunks(
                args,
                kwargs,
                self._n_microbatches,
                args_chunk_spec,
                kwargs_chunk_spec,
            )
            return args_split, kwargs_split
        else:
            # Empty inputs (e.g. when called on middle stages)
            # Return a list of empty tuples/dicts with matching length as chunks
            return [()] * self._n_microbatches, [{}] * self._n_microbatches

    def _merge_outputs(self, output_chunks: List[Any]) -> Any:
        """
        Merge output chunks back to a batch state.
        If output_merge_spec is None, the utility will merge output chunks by dimension 0 (batch dim).
        """
        return merge_chunks(
            output_chunks,
            self._output_merge_spec,
        )


def _batch_p2p(p2p_ops: List[dist.P2POp], desc: Optional[str] = None):
    """
    Simple wrapper over batch_isend_irecv from torch.distributed, which just adds a descriptive logger on top.
    """
    if len(p2p_ops) == 0:
        return None
    desc_str = f"{desc}, " if desc else ""
    logger.debug(f"batch_p2p {desc_str}{p2p_ops}")  # noqa: G004
    return dist.batch_isend_irecv(p2p_ops).pop()


def _sorted_batch_p2p(
    p2p_ops: List[dist.P2POp], desc: Optional[str] = None
) -> Dict[int, dist.Work]:
    """
    Sorts the list of P2P ops by the peer rank, and then calls
    batch_isend_irecv. Return a dictionary of works by peer rank. This function
    helps us avoid hangs in case of skip connections.
    """
    # Arrange p2p_ops by peer rank:
    #   int is the peer rank;
    #   List is the list of ops towards the peer
    ops_by_peer: Dict[int, List[dist.P2POp]] = defaultdict(list)
    work_by_peer: Dict[int, dist.Work] = {}
    if len(p2p_ops) == 0:
        return work_by_peer

    # Classify the ops by peer rank
    for op in p2p_ops:
        ops_by_peer[op.peer].append(op)

    # Call batch_isend_irecv per peer, in sorted order of the peers (to avoid hangs)
    for peer, ops in sorted(ops_by_peer.items()):
        work_by_peer[peer] = _batch_p2p(ops, desc=desc)

    return work_by_peer


class PipelineScheduleSingle(_PipelineSchedule):
    """
    Base class for single-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.
    """

    def __init__(
        self,
        stage: _PipelineStageBase,
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )
        self._pipe_info = (
            stage.pipe_info if hasattr(stage, "pipe_info") else None  # type: ignore[attr-defined]
        )
        # Self attributes
        self._stage = stage
        self._num_stages = stage.num_stages
        # Set the same has_backward flag for stage object
        self._stage.has_backward = self._has_backward

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        # Clean per iteration
        self._stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        if self._stage.is_last:
            return self._merge_outputs(self._stage.output_chunks)
        else:
            return None


class ScheduleGPipe(PipelineScheduleSingle):
    """
    The GPipe schedule.
    Will go through all the microbatches in a fill-drain manner.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the GPipe schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # Delay send waits
        fwd_sends_to_wait: List[dist.Work] = []

        # Run microbatches
        for i in range(self._n_microbatches):
            with record_function(f"Forward {i}"):
                ops = self._stage.get_fwd_recv_ops()
                works = _sorted_batch_p2p(ops, desc="fwd_recv")
                for work in works.values():
                    work.wait()

                output = self._stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])  # type: ignore[index]

                ops = self._stage.get_fwd_send_ops()
                works = _sorted_batch_p2p(ops, desc="fwd_send")
                fwd_sends_to_wait.extend(works.values())

            logger.debug(
                f"[{self._stage.stage_index}] Forwarded microbatch {i}"  # noqa: G004
            )

            self._maybe_compute_loss(self._stage, output, target_mbs, i)

        # Wait for all forward sends to finish
        # This should not have performance impact because by the time the first
        # backward arrives all the forward sends should have been finished.
        for work in fwd_sends_to_wait:
            work.wait()

        # No loss function, no need to run backward
        if not self._has_backward:
            return

        # Run backward
        # Delay send waits
        bwd_sends_to_wait: List[dist.Work] = []
        for i in range(self._n_microbatches):
            # set library-specific data-parallel config flags to ensure gradient accumulation across microbatches
            self._stage._configure_data_parallel_mode(i == self._n_microbatches - 1)

            with record_function(f"Backward {i}"):
                ops = self._stage.get_bwd_recv_ops()
                works = _sorted_batch_p2p(ops, desc="bwd_recv")
                for work in works.values():
                    work.wait()

                loss = self._maybe_get_loss(self._stage, i)
                self._stage.backward_one_chunk(loss=loss)

                ops = self._stage.get_bwd_send_ops()
                works = _sorted_batch_p2p(ops, desc="bwd_send")
                bwd_sends_to_wait.extend(works.values())

            logger.debug(
                f"[{self._stage.stage_index}] Backwarded microbatch {i}"  # noqa: G004
            )

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)

        # Wait for all backward sends to finish
        for work in bwd_sends_to_wait:
            work.wait()


class Schedule1F1B(PipelineScheduleSingle):
    """
    The 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady state.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the 1F1B schedule.

        Args:
            microbatches: list of microbatch args.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # Last stage has 1 warmup, second-to-last 2 warmups, ...
        # first stage `num_stages` warmups
        warmup_chunks = min(
            self._n_microbatches,
            self._num_stages - self._stage.stage_index,
        )

        # Chunk counters
        fwd_mb_index = 0
        bwd_mb_index = 0

        # Warmup phase
        send_work = None
        fwd_sends = []
        for _ in range(warmup_chunks):
            # Receive activations
            fwd_recvs = self._stage.get_fwd_recv_ops()
            if recv_work := _batch_p2p(fwd_recvs, desc="fwd_recv"):
                recv_work.wait()

            # Compute
            output = self._stage.forward_one_chunk(arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]

            # Clear previous chunk's forward sends (hopefully they have well
            # finished, otherwise, we are heavily communication bound, in which
            # case it doesn't create a lot of benefit to compute next chunk
            # eagerly either)
            if send_work:
                send_work.wait()

            # Send activations
            fwd_sends = self._stage.get_fwd_send_ops()
            if fwd_mb_index != warmup_chunks - 1:
                # Safe to fire
                send_work = _batch_p2p(fwd_sends, desc="fwd_send")
            # otherwise:
            #   The last foward send is left for fuse with first 1B in 1B1F below

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)
            fwd_mb_index += 1

        # Now we should have send ops left over, to be fused with first 1B of 1B1F phase below.

        # 1B1F phase
        while True:  # Don't worry, we have a break inside
            # We actually do 1B first as the `1B1F` name indicates, so prepare its recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops()

            # Now, we need to fire the fwd_sends and bwd_recvs together
            if fuse_work := _batch_p2p(fwd_sends + bwd_recvs, desc="fwd_send_bwd_recv"):
                fuse_work.wait()

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(loss=loss)

            # Get the bwd send ops, but don't fire, to be fused with the 1F below
            bwd_sends = self._stage.get_bwd_send_ops()
            bwd_mb_index += 1

            if fwd_mb_index == self._n_microbatches:
                # We are done with 1B1F, so break with some left-over bwd_sends
                break

            # We prepare 1F of the `1B1F`
            fwd_recvs = self._stage.get_fwd_recv_ops()

            # Fuse it with bwd_sends above
            if fuse_work := _batch_p2p(bwd_sends + fwd_recvs, desc="bwd_send_fwd_recv"):
                fuse_work.wait()

            # Now do the fwd
            output = self._stage.forward_one_chunk(arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]

            # Compute loss
            self._maybe_compute_loss(self._stage, output, target_mbs, fwd_mb_index)

            # Get the fwd send ops, but don't fire, leave it for the next iter (wrap-around)
            fwd_sends = self._stage.get_fwd_send_ops()
            fwd_mb_index += 1

        # Remember we still have some bwd_sends left over after the break? Now it is time to fire it
        send_work = _batch_p2p(bwd_sends, desc="bwd_send")

        # Cooldown
        while bwd_mb_index < self._n_microbatches:
            # prepare bwd recv ops
            bwd_recvs = self._stage.get_bwd_recv_ops()
            if recv_work := _batch_p2p(bwd_recvs, desc="bwd_recv"):
                recv_work.wait()

            # Backward one chunk
            loss = self._maybe_get_loss(self._stage, bwd_mb_index)
            self._stage.backward_one_chunk(loss=loss)

            # Clear previous chunk's backward sends (hopefully they have well finished)
            if send_work:
                send_work.wait()

            # Get the bwd send ops, fire it
            bwd_sends = self._stage.get_bwd_send_ops()
            send_work = _batch_p2p(bwd_sends, desc="bwd_send")
            bwd_mb_index += 1

        # Wait for the last backward send to finish
        if send_work:
            send_work.wait()

        # Return losses if there is a container passed in
        self._update_losses(self._stage, losses)


class PipelineScheduleMulti(_PipelineSchedule):
    """
    Base class for multi-stage schedules.
    Implements the `step` method.
    Derived classes should implement `_step_microbatches`.
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        if len(stages) <= 1:
            raise ValueError(
                f"Multi-stage schedule expects at least two stages but got {len(stages)}"
            )
        # Init parent
        super().__init__(
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )
        self._pipe_info = (
            stages[0].pipe_info if hasattr(stages[0], "pipe_info") else None  # type: ignore[attr-defined]
        )
        # Self attributes
        self._stages = stages
        self._num_stages = stages[0].num_stages
        # Set the same has_backward flag for stage object
        for stage in self._stages:
            stage.has_backward = self._has_backward

        self._should_compute_loss = (
            lambda stage: stage.is_last and self._loss_fn is not None
        )

    def step(self, *args, target=None, losses: Optional[List] = None, **kwargs):
        # Clean per iteration
        for stage in self._stages:
            stage.clear_runtime_states()

        # Split inputs into microbatches
        args_split, kwargs_split = self._split_inputs(args, kwargs)

        # Split target into microbatches
        if target is not None:
            targets_split = list(torch.tensor_split(target, self._n_microbatches))
        else:
            targets_split = None

        # Run microbatches
        self._step_microbatches(args_split, kwargs_split, targets_split, losses)

        # Return merged results per original format
        for stage in self._stages:
            if stage.is_last:
                return self._merge_outputs(stage.output_chunks)
        # Does not contain the last stage
        return None


class ScheduleLoopedBFS(PipelineScheduleMulti):
    """
    Breadth-First Pipeline Parallelism.
    See https://arxiv.org/abs/2211.05953 for details.
    Simliar to Interleaved 1F1B, Looped BFS supports multiple stages per rank.
    What is different is that when microbatches are ready for multiple local
    stages, Loops BFS will prioritizes the earlier stage, running all available
    microbatches at once.
    """

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,  # TODO
        losses: Optional[List] = None,  # TODO
    ):
        """
        Run one iteration of the pipeline schedule with list of microbatches.
        Will go through all the microbatches according to the Looped BFS schedule.

        Args:
            microbatches: list of microbatch args.
        """
        # Pre-process inputs
        if arg_mbs is not None:
            # TODO: fix this so it is preset
            self._n_microbatches = len(arg_mbs)
            assert len(arg_mbs) == self._n_microbatches
        else:
            arg_mbs = [()] * self._n_microbatches

        if kwarg_mbs is not None:
            assert len(kwarg_mbs) == self._n_microbatches
        else:
            kwarg_mbs = [{}] * self._n_microbatches

        for stage in self._stages:
            for i in range(self._n_microbatches):
                with record_function(f"Stage {stage.stage_index} Forward"):
                    ops = stage.get_fwd_recv_ops()
                    if ops:
                        _batch_p2p(ops, desc="fwd_recv").wait()

                    output = stage.forward_one_chunk(arg_mbs[i], kwarg_mbs[i])
                    self._maybe_compute_loss(stage, output, target_mbs, i)

                    ops = stage.get_fwd_send_ops()
                    if ops:
                        _batch_p2p(ops, desc="fwd_send")

        for stage in reversed(self._stages):
            for i in range(self._n_microbatches):
                stage._configure_data_parallel_mode(i == self._n_microbatches - 1)
                with record_function(f"Stage {stage.stage_index} Backward"):
                    ops = stage.get_bwd_recv_ops()
                    if ops:
                        _batch_p2p(ops, desc="bwd_recv").wait()

                    loss = self._maybe_get_loss(stage, i)
                    stage.backward_one_chunk(loss=loss)

                    ops = stage.get_bwd_send_ops()
                    if ops:
                        _batch_p2p(ops, desc="bwd_send")

        self._update_losses(self._stages, losses)


class ScheduleInterleaved1F1B(PipelineScheduleMulti):
    """
    The Interleaved 1F1B schedule.
    Will perform one forward and one backward on the microbatches in steady
    state and supports multiple stages per rank. When microbatches are ready for
    multiple local stages, Interleaved 1F1B prioritizes the earlier microbatch
    (also called "depth first").
    """

    def __init__(
        self,
        stages: List[_PipelineStageBase],
        n_microbatches: int,
        loss_fn: Optional[Callable] = None,
        output_merge_spec: Optional[Union[Dict[str, Any], Tuple[Any]]] = None,
    ):
        self.pp_group_size = stages[0].group_size
        # TODO: is this limitation a must?
        if n_microbatches % self.pp_group_size != 0:
            raise ValueError(
                "Interleaved 1F1B requires the number of microbatches to be a "
                f"multiple of the number of pipeline ranks ({self.pp_group_size}), "
                f"but got {n_microbatches}."
            )

        super().__init__(
            stages=stages,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            output_merge_spec=output_merge_spec,
        )

        self.n_local_stages = len(stages)
        self.rank = stages[0].group_rank

    def _step_microbatches(
        self,
        arg_mbs: Optional[List] = None,
        kwarg_mbs: Optional[List] = None,
        target_mbs: Optional[List] = None,
        losses: Optional[List] = None,
    ):
        """
        Operate on the microbatches for interleaved 1f1b schedule (https://arxiv.org/pdf/2104.04473.pdf).

        Highest rank has a warmup (fwd only) count of [len(stages) - 1] * number of PP ranks
        and each rank away from highest rank adds 2 warmup steps due to:
            - one happened before highest rank's warmup started,
            - one waiting for backward result to trickle down from highest rank

        TODO: Interleaved 1F1B does not support using _sorted_batch_p2p()
        because it requires recvs and sends from different peers
        to execute in the same coalesced operation. As a result, this schedule does
        not support models with skip connections.
        """
        arg_mbs, kwarg_mbs = self._check_inputs(arg_mbs, kwarg_mbs, target_mbs, losses)

        # increment warmup_steps by 2 for each hop away
        warmup_steps = (self.n_local_stages - 1) * self.pp_group_size
        warmup_steps += 2 * ((self.pp_group_size - 1) - self.rank)
        warmup_steps = min(warmup_steps, self._n_microbatches * self.n_local_stages)
        fwd_bwd_steps = (self.n_local_stages * self._n_microbatches) - warmup_steps
        cooldown_steps = (self.n_local_stages * self._n_microbatches) - fwd_bwd_steps

        assert (
            warmup_steps + fwd_bwd_steps * 2 + cooldown_steps
            == self.n_local_stages * self._n_microbatches * 2
        )
        total_steps = warmup_steps + fwd_bwd_steps + cooldown_steps

        logger.debug(
            f"rank {self.rank}, warmup_steps {warmup_steps}, "  # noqa: G004
            f"1f1b {fwd_bwd_steps}, cooldown_steps {cooldown_steps}"
        )

        def forward_stage_local_index(step):
            return (step // self.pp_group_size) % self.n_local_stages

        def backward_stage_local_index(step):
            return (
                self.n_local_stages
                - 1
                - ((step - warmup_steps) // self.pp_group_size) % self.n_local_stages
            )

        fwd_stage_mb_index: Dict[_PipelineStageBase, int] = defaultdict(int)
        bwd_stage_mb_index: Dict[_PipelineStageBase, int] = defaultdict(int)

        # Delay send waits
        sends_to_wait: List[dist.Work] = []

        # Store ops (potentially across steps)
        ops: List[dist.P2POp] = []

        # Warmup Phase (forward only)
        for step in range(warmup_steps):
            fwd_stage = self._stages[forward_stage_local_index(step)]

            # This will assign the current microbatch index and update it for future steps
            fwd_stage_mb_index[fwd_stage] = (
                mb_index := fwd_stage_mb_index[fwd_stage]
            ) + 1

            logger.debug(
                f"Rank {self.rank}: {step=}, {fwd_stage.stage_index=}, {mb_index=}"  # noqa: G004
            )

            with record_function(f"Forward {step}"):
                ops.extend(fwd_stage.get_fwd_recv_ops())
                if ops:
                    work = _batch_p2p(ops, desc="warmup_pre_fwd")
                    work.wait()
                    ops.clear()

                output = fwd_stage.forward_one_chunk(arg_mbs[mb_index], kwarg_mbs[mb_index])  # type: ignore[index]

                ops.extend(fwd_stage.get_fwd_send_ops())
                # If we are right before the fwd-bwd step, then we need to delay the send to the next step,
                # This is because fwd-bwd send/recvs among ranks need to be aligned to prevent a hang.
                # In the edge cases where there are no fwd_bwds and cooldown is immediate, then no delay is needed
                if ops and (step != warmup_steps - 1 or fwd_bwd_steps == 0):
                    work = _batch_p2p(ops, desc="warmup_post_fwd")
                    sends_to_wait.append(work)
                    ops.clear()

                self._maybe_compute_loss(fwd_stage, output, target_mbs, mb_index)

        # 1F1B Phase (forward and backward)
        for step in range(warmup_steps, warmup_steps + fwd_bwd_steps):
            fwd_stage = self._stages[forward_stage_local_index(step)]
            bwd_stage = self._stages[backward_stage_local_index(step)]

            fwd_stage_mb_index[fwd_stage] = (
                fwd_mb_index := fwd_stage_mb_index[fwd_stage]
            ) + 1
            bwd_stage_mb_index[bwd_stage] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage]
            ) + 1

            bwd_stage._configure_data_parallel_mode(
                bwd_mb_index == self._n_microbatches - 1
            )
            logger.debug(
                f"Rank {self.rank}: {step=}, {fwd_stage.stage_index=}, "  # noqa: G004
                f"{bwd_stage.stage_index=}, {fwd_mb_index=}, {bwd_mb_index=}"
            )
            desc = f"1F1B {step}"
            with record_function(desc):
                ops.extend(fwd_stage.get_fwd_recv_ops())
                ops.extend(bwd_stage.get_bwd_recv_ops())
                if ops:
                    work = _batch_p2p(ops, desc=desc)
                    work.wait()
                    ops.clear()

                # Forward
                output = fwd_stage.forward_one_chunk(arg_mbs[fwd_mb_index], kwarg_mbs[fwd_mb_index])  # type: ignore[index]
                ops.extend(fwd_stage.get_fwd_send_ops())
                self._maybe_compute_loss(fwd_stage, output, target_mbs, fwd_mb_index)

                # Backward
                loss = self._maybe_get_loss(bwd_stage, bwd_mb_index)
                bwd_stage.backward_one_chunk(loss=loss)
                ops.extend(bwd_stage.get_bwd_send_ops())

        # Cooldown Phase (backward only)
        for step in range(warmup_steps + fwd_bwd_steps, total_steps):
            bwd_stage = self._stages[backward_stage_local_index(step)]
            bwd_stage_mb_index[bwd_stage] = (
                bwd_mb_index := bwd_stage_mb_index[bwd_stage]
            ) + 1
            bwd_stage._configure_data_parallel_mode(
                bwd_mb_index == self._n_microbatches - 1
            )

            logger.debug(
                f"Rank {self.rank}: {step=}, {bwd_stage.stage_index=}, {bwd_mb_index=}"  # noqa: G004
            )
            desc = f"Cooldown {step}"
            with record_function(desc):
                ops.extend(bwd_stage.get_bwd_recv_ops())
                if ops:
                    work = _batch_p2p(ops, desc=desc + " pre_bwd")
                    work.wait()
                    ops.clear()

                loss = self._maybe_get_loss(bwd_stage, bwd_mb_index)
                bwd_stage.backward_one_chunk(loss=loss)

                ops.extend(bwd_stage.get_bwd_send_ops())
                if ops:
                    work = _batch_p2p(ops, desc=desc + " post_bwd")
                    sends_to_wait.append(work)
                    ops.clear()

        # Make sure all sends are finished
        for work in sends_to_wait:
            work.wait()

        # Return losses if there is a container passed in
        self._update_losses(self._stages, losses)
