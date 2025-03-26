import inspect
import itertools
from typing import Callable, List

import cuda.parallel.experimental.algorithms as algorithms
import torch
import torch.utils._pytree as pytree


function_registry = {}


def create_input_slices(xs: torch.Tensor, dim: int) -> List[List[slice | int]]:
    """Given a multidimensional input tensor and a dimension to run the scan
    over, we want to slice it so that we can call parallel scan on each sliced
    input array individually. This function generates the slices, which are
    lists of slice objects and ints, that can be used to index the input and
    output array."""

    assert dim < xs.dim()
    dim_shape = xs.shape[dim]

    # e.g., for a tensor of shape [3, 2, 4], and for dim == 0, this will be
    # [range(2), range(4)]
    range_objects = [torch.arange(i) for idx, i in enumerate(xs.shape) if idx != dim]

    # We now create the slice objects to slice the array by getting every
    # combination of elements. For the dimension we are scanning over, we insert
    # a slice that takes that entire dimension. Following the example above, we
    # get: [[slice(0, 3), 0, 0], [slice(0, 3), 0, 1], ..., [slice(0, 3), 1, 3]]
    # which is of size 2x4
    slice_objects = []
    for s in itertools.product(*range_objects):
        current_slice = list(s)
        current_slice.insert(dim, slice(0, dim_shape))
        slice_objects.append(current_slice)

    return slice_objects


@torch.library.custom_op("cccl::inclusive_scan", mutates_args=())
def inclusive_scan(combine_fn_name: str, xs: torch.Tensor, dim: int) -> torch.Tensor:
    slice_objects = create_input_slices(xs, dim)
    d_output = torch.empty_like(xs)

    for slice_object in slice_objects:
        current_output = d_output[*slice_object]
        current_input = xs[*slice_object]

        # TODO: flatten is potentially expensive
        h_init = torch.tensor([torch.flatten(current_input)[0]], dtype=xs.dtype).numpy()

        combine_fn = function_registry[combine_fn_name]

        # Instantiate scan for the given operator and initial value
        scanner = algorithms.inclusive_scan(d_output, d_output, combine_fn, h_init)

        input_array = current_input[1:]
        output_array = current_output[1:]
        size = xs.size(dim) - 1

        # Determine temporary device storage requirements
        temp_storage_size = scanner(None, input_array, output_array, size, h_init)

        # Allocate temporary storage
        d_temp_storage = torch.empty((temp_storage_size,), dtype=torch.uint8).cuda()

        # Run reduction
        scanner(d_temp_storage, input_array, output_array, size, h_init)

        current_output.data[0] = h_init.item()

    return d_output


@inclusive_scan.register_fake
def _(combine_fn_name, xs, dim):
    return torch.empty_like(xs)


def wrap_if_lambda(func: Callable) -> Callable:
    """Transform a lambda operator into a normal function"""

    if not callable(func):
        raise TypeError("Input must be callable")

    sig = inspect.signature(func)
    args: str = ", ".join(sig.parameters.keys())
    name = f"wrapped_{id(func)}"

    # TODO: we still need to get the body of the lambda. Could this be improved?
    # inspect.getsource(func) doesn't work if there is other code on the same
    # line
    positions = list(func.__code__.co_positions())[-1]
    body_line = positions[0]
    body_start = positions[2]
    body_end = positions[3]
    file = inspect.getsourcefile(func)

    with open(file, "r") as f:
        line = f.readlines()[body_line - 1]
        body = line[body_start:body_end]

    # This is how we handle torch.abs()
    body = body.replace("torch.", "")

    function_str = f"""
def {name}({args}):
    return {body}
"""

    # This will store the function definition in this dictionary
    namespace = {}
    exec(function_str, namespace)

    return namespace[name]


def associative_scan(
    combine_fn: Callable[[pytree.PyTree, pytree.PyTree], pytree.PyTree],
    xs: pytree.PyTree,
    dim: int,
    reverse: bool = False,
    combine_mode: str = "pointwise",
) -> torch.Tensor:
    r"""
    Performs an inclusive scan with an associative combine function.

    .. warning::
        `torch.associative_scan` is a prototype feature in PyTorch. It currently
        does not support autograd and you may run into miscompiles.
        Read more about feature classification at:
        https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    This operator requires runtime code generation and so requires support for
    ``torch.compile``. Further, only CUDA device codegen is supported at the moment.

    Args:
        combine_fn (Callable): A binary callable with type ``(Tensor, Tensor) -> Tensor``,
            or if input is a pytree ``(pytree, pytree) -> pytree``.
            This function must be pure, i.e., no lifted arguments are supported at the moment,
            satisfy the associative property and have no side-effects.
        xs (torch.Tensor): The input tensor, or nested pytree of tensors.
            All inputs are expected to have the same shape.
        dim (int): the dimension to scan over
        reverse (bool): A boolean stating if the scan should be reversed with respect to ``dim``, default ``False``.
        combine_mode (str): A string indicating whether the ``combine_fn`` is ``pointwise`` or ``generic``, default ``pointwise``.
            If ``combine_mode=pointwise``, ``combine_fn`` must be pure, may only contain pointwise operations
            and ``xs`` must be CUDA tensors.
            In all other cases ``combine_mode=generic`` should be used.
            Note: ``combine_mode=pointwise`` is more efficient than ``combine_mode=generic``.


    Example::

        def add(x: torch.Tensor, y: torch.Tensor):
            return x + y

        cumsum = associative_scan(add, x, dim)

    """
    if not callable(combine_fn):
        raise ValueError("Combine_fn must be a callable, but got {combine_fn}")
    if not isinstance(dim, int):
        raise ValueError("Dim must be an int, but got " + str(type(dim)))
    if combine_mode not in ["pointwise", "generic"]:
        raise ValueError(
            "Combine_mode must either 'pointwise' or 'generic', but got {combine_mode}"
        )

    if combine_mode == "generic":
        raise ValueError("cuda_parallel.associative_scan does not currently support \"generic\"")

    # TODO: cuda.parallel doesn't support lambdas
    if combine_fn.__name__ == "<lambda>":
        combine_fn = wrap_if_lambda(combine_fn)

    # TODO: instead of using the name, is there a better unique identifier? What
    # if there is a naming clash?
    combine_fn_name = combine_fn.__name__
    function_registry[combine_fn_name] = combine_fn

    return inclusive_scan(combine_fn_name, xs, dim)
