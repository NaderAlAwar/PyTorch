from typing import Callable

import cuda.parallel.experimental.algorithms as algorithms
import numpy as np
import torch
import torch.utils._pytree as pytree


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

    h_init = torch.tensor([0], dtype=xs.dtype).numpy()
    d_output = torch.empty_like(xs)

    # Instantiate scan for the given operator and initial value
    scanner = algorithms.inclusive_scan(d_output, d_output, combine_fn, h_init)

    # Determine temporary device storage requirements
    temp_storage_size = scanner(None, xs, d_output, xs.size(dim), h_init)

    # Allocate temporary storage
    d_temp_storage = torch.empty((temp_storage_size,), dtype=torch.uint8).cuda()

    # Run reduction
    scanner(d_temp_storage, xs, d_output, xs.size(dim), h_init)

    return d_output
