from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import warnings


def detach_variable(inputs):
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ", type(inputs).__name__)


def check_backward_validity(inputs):
    if not any(inp.requires_grad for inp in inputs):
        warnings.warn("None of the inputs have requires_grad=True. Gradients will be None")


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        if preserve_rng_state:
            # We can't know if the user will transfer some args from the host
            # to the device during their run_fn.  Therefore, we stash both
            # the cpu and cuda rng states unconditionally.
            #
            # TODO:
            # We also can't know if the run_fn will internally move some args to a device
            # other than the current device, which would require logic to preserve
            # rng states for those devices as well...but I see no easy way to
            # handle such cases.
            ctx.fwd_cpu_rng_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpointing is not compatible with .grad(), please use .backward() if possible")
        inputs = ctx.saved_tensors
        if ctx.preserve_rng_state:
            # Stash the surrounding rng state, and mimic the state that was
            # present at this time during forward.  If cuda was not initialized
            # at this time during the original forward, we assume that the imminent
            # forward-segment-within-backward will not alter the cuda rng state.
            current_cpu_rng_state = torch.get_rng_state()
            torch.set_rng_state(ctx.fwd_cpu_rng_state)
            if ctx.had_cuda_in_fwd:
                current_cuda_rng_state = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(ctx.fwd_cuda_rng_state)
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)
        if ctx.preserve_rng_state:
            # Restore the surrounding rng state
            torch.set_rng_state(current_cpu_rng_state)
            if ctx.had_cuda_in_fwd:
                torch.cuda.set_rng_state(current_cuda_rng_state)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        torch.autograd.backward(outputs, args)
        return (None, None) + tuple(inp.grad for inp in detached_inputs)


def checkpoint(function, *args, preserve_rng_state=False):
    r"""Checkpoint a model or part of the model

    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.

    Specifically, in the forward pass, :attr:`function` will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retreived, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.

    .. warning::
        Checkpointing doesn't work with :func:`torch.autograd.grad`, but only
        with :func:`torch.autograd.backward`.

    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    .. warning:
        Checkpointing is implemented by rerunning a forward-pass segment for
        each checkpointed segment during backward.  This can result in running
        states like the RNG state used for dropout to be advanced more than
        they would be without checkpointing, which can cause checkpoints that
        include dropout invocations to have non-deterministic output
        compared to non-checkpointed passes.  Use ``preserve_rng_state`` if
        bitwise accuracy is desired.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        args: tuple containing inputs to the :attr:`function`
        preserve_rng_state (bool, optional, default=False): If ``True``, stashes
            and restores the RNG state such that checkpointed segments making use of
            the RNG state (via e.g. dropout) are bitwise accurate with non-checkpointed passes.
            This can incur a moderate performance hit depending on the runtime of each segment.

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    return CheckpointFunction.apply(function, preserve_rng_state, *args)


def checkpoint_sequential(functions, segments, *inputs, preserve_rng_state=False):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. The inputs of each checkpointed segment will be saved for
    re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing doesn't work with :func:`torch.autograd.grad`, but only
        with :func:`torch.autograd.backward`.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        inputs: tuple of Tensors that are inputs to :attr:`functions`
        preserve_rng_state (bool, optional, default=False): If ``True``, stashes
            and restores the RNG state such that checkpointed segments making use of
            the RNG state (via e.g. dropout) are bitwise accurate with non-checkpointed passes.
            This can incur a moderate performance hit depending on the runtime of each segment.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """

    def run_function(start, end, functions):
        def forward(*inputs):
            input = inputs[0]
            for j in range(start, end + 1):
                input = functions[j](input)
            return input
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = len(functions) // segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, segment_size * (segments - 1), segment_size):
        end = start + segment_size - 1
        inputs = checkpoint(run_function(start, end, functions), *inputs,
                            preserve_rng_state=preserve_rng_state)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
    return run_function(end + 1, len(functions) - 1, functions)(*inputs)
