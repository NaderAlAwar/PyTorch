import sys

import torch
from torch._C import _add_docstr, _linalg  # type: ignore

Tensor = torch.Tensor

# Note: This not only adds doc strings for functions in the linalg namespace, but
# also connects the torch.linalg Python namespace to the torch._C._linalg builtins.

cholesky = _add_docstr(_linalg.linalg_cholesky, r"""
linalg.cholesky(input, *, out=None) -> Tensor

Computes the Cholesky decomposition of a Hermitian (or symmetric for real-valued matrices)
positive-definite matrix or the Cholesky decompositions for a batch of such matrices.
Each decomposition has the form:

.. math::

    \text{input} = LL^H

where :math:`L` is a lower-triangular matrix and :math:`L^H` is the conjugate transpose of :math:`L`,
which is just a transpose for the case of real-valued input matrices.
In code it translates to ``input = L @ L.t()` if :attr:`input` is real-valued and
``input = L @ L.conj().t()`` if :attr:`input` is complex-valued.
The batch of :math:`L` matrices is returned.

Supports real-valued and complex-valued inputs.

.. note:: If :attr:`input` is not a Hermitian positive-definite matrix, or if it's a batch of matrices
          and one or more of them is not a Hermitian positive-definite matrix, then a RuntimeError will be thrown.
          If :attr:`input` is a batch of matrices, then the error message will include the batch index
          of the first matrix that is not Hermitian positive-definite.

.. warning:: This function always checks whether :attr:`input` is a Hermitian positive-definite matrix
             using `info` argument to LAPACK/MAGMA call. For CUDA this causes cross-device memory synchronization.

Args:
    input (Tensor): the input tensor of size :math:`(*, n, n)` consisting of Hermitian positive-definite
                    :math:`n \times n` matrices, where `*` is zero or more batch dimensions.

Keyword args:
    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

Examples::

    >>> a = torch.randn(2, 2, dtype=torch.complex128)
    >>> a = torch.mm(a, a.t().conj())  # creates a Hermitian positive-definite matrix
    >>> l = torch.linalg.cholesky(a)
    >>> a
    tensor([[2.5266+0.0000j, 1.9586-2.0626j],
            [1.9586+2.0626j, 9.4160+0.0000j]], dtype=torch.complex128)
    >>> l
    tensor([[1.5895+0.0000j, 0.0000+0.0000j],
            [1.2322+1.2976j, 2.4928+0.0000j]], dtype=torch.complex128)
    >>> torch.mm(l, l.t().conj())
    tensor([[2.5266+0.0000j, 1.9586-2.0626j],
            [1.9586+2.0626j, 9.4160+0.0000j]], dtype=torch.complex128)

    >>> a = torch.randn(3, 2, 2, dtype=torch.float64)
    >>> a = torch.matmul(a, a.transpose(-2, -1))  # creates a symmetric positive-definite matrix
    >>> l = torch.linalg.cholesky(a)
    >>> a
    tensor([[[ 1.1629,  2.0237],
            [ 2.0237,  6.6593]],

            [[ 0.4187,  0.1830],
            [ 0.1830,  0.1018]],

            [[ 1.9348, -2.5744],
            [-2.5744,  4.6386]]], dtype=torch.float64)
    >>> l
    tensor([[[ 1.0784,  0.0000],
            [ 1.8766,  1.7713]],

            [[ 0.6471,  0.0000],
            [ 0.2829,  0.1477]],

            [[ 1.3910,  0.0000],
            [-1.8509,  1.1014]]], dtype=torch.float64)
    >>> torch.allclose(torch.matmul(l, l.transpose(-2, -1)), a)
    True
""")

det = _add_docstr(_linalg.linalg_det, r"""
linalg.det(input) -> Tensor

Alias of :func:`torch.det`.
""")

norm = _add_docstr(_linalg.linalg_norm, r"""
linalg.norm(input, ord=None, dim=None, keepdim=False, *, out=None, dtype=None) -> Tensor

Returns the matrix norm or vector norm of a given tensor.

This function can calculate one of eight different types of matrix norms, or one
of an infinite number of vector norms, depending on both the number of reduction
dimensions and the value of the `ord` parameter.

Args:
    input (Tensor): The input tensor. If dim is None, x must be 1-D or 2-D, unless :attr:`ord`
        is None. If both :attr:`dim` and :attr:`ord` are None, the 2-norm of the input flattened to 1-D
        will be returned.

    ord (int, float, inf, -inf, 'fro', 'nuc', optional): The order of norm.
        inf refers to :attr:`float('inf')`, numpy's :attr:`inf` object, or any equivalent object.
        The following norms can be calculated:

        =====  ============================  ==========================
        ord    norm for matrices             norm for vectors
        =====  ============================  ==========================
        None   Frobenius norm                2-norm
        'fro'  Frobenius norm                -- not supported --
        'nuc'  nuclear norm                  -- not supported --
        inf    max(sum(abs(x), dim=1))       max(abs(x))
        -inf   min(sum(abs(x), dim=1))       min(abs(x))
        0      -- not supported --           sum(x != 0)
        1      max(sum(abs(x), dim=0))       as below
        -1     min(sum(abs(x), dim=0))       as below
        2      2-norm (largest sing. value)  as below
        -2     smallest singular value       as below
        other  -- not supported --           sum(abs(x)**ord)**(1./ord)
        =====  ============================  ==========================

        Default: ``None``

    dim (int, 2-tuple of ints, 2-list of ints, optional): If :attr:`dim` is an int,
        vector norm will be calculated over the specified dimension. If :attr:`dim`
        is a 2-tuple of ints, matrix norm will be calculated over the specified
        dimensions. If :attr:`dim` is None, matrix norm will be calculated
        when the input tensor has two dimensions, and vector norm will be
        calculated when the input tensor has one dimension. Default: ``None``

    keepdim (bool, optional): If set to True, the reduced dimensions are retained
        in the result as dimensions with size one. Default: ``False``

Keyword args:

    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

    dtype (:class:`torch.dtype`, optional): If specified, the input tensor is cast to
        :attr:`dtype` before performing the operation, and the returned tensor's type
        will be :attr:`dtype`. If this argument is used in conjunction with the
        :attr:`out` argument, the output tensor's type must match this argument or a
        RuntimeError will be raised. Default: ``None``

Examples::

    >>> import torch
    >>> from torch import linalg as LA
    >>> a = torch.arange(9, dtype=torch.float) - 4
    >>> a
    tensor([-4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])
    >>> b = a.reshape((3, 3))
    >>> b
    tensor([[-4., -3., -2.],
            [-1.,  0.,  1.],
            [ 2.,  3.,  4.]])

    >>> LA.norm(a)
    tensor(7.7460)
    >>> LA.norm(b)
    tensor(7.7460)
    >>> LA.norm(b, 'fro')
    tensor(7.7460)
    >>> LA.norm(a, float('inf'))
    tensor(4.)
    >>> LA.norm(b, float('inf'))
    tensor(9.)
    >>> LA.norm(a, -float('inf'))
    tensor(0.)
    >>> LA.norm(b, -float('inf'))
    tensor(2.)

    >>> LA.norm(a, 1)
    tensor(20.)
    >>> LA.norm(b, 1)
    tensor(7.)
    >>> LA.norm(a, -1)
    tensor(0.)
    >>> LA.norm(b, -1)
    tensor(6.)
    >>> LA.norm(a, 2)
    tensor(7.7460)
    >>> LA.norm(b, 2)
    tensor(7.3485)

    >>> LA.norm(a, -2)
    tensor(0.)
    >>> LA.norm(b.double(), -2)
    tensor(1.8570e-16, dtype=torch.float64)
    >>> LA.norm(a, 3)
    tensor(5.8480)
    >>> LA.norm(a, -3)
    tensor(0.)

Using the :attr:`dim` argument to compute vector norms::

    >>> c = torch.tensor([[1., 2., 3.],
    ...                   [-1, 1, 4]])
    >>> LA.norm(c, dim=0)
    tensor([1.4142, 2.2361, 5.0000])
    >>> LA.norm(c, dim=1)
    tensor([3.7417, 4.2426])
    >>> LA.norm(c, ord=1, dim=1)
    tensor([6., 6.])

Using the :attr:`dim` argument to compute matrix norms::

    >>> m = torch.arange(8, dtype=torch.float).reshape(2, 2, 2)
    >>> LA.norm(m, dim=(1,2))
    tensor([ 3.7417, 11.2250])
    >>> LA.norm(m[0, :, :]), LA.norm(m[1, :, :])
    (tensor(3.7417), tensor(11.2250))
""")

tensorinv = _add_docstr(_linalg.linalg_tensorinv, r"""
linalg.tensorinv(input, ind=2, *, out=None) -> Tensor

Computes a tensor ``input_inv`` such that ``tensordot(input_inv, input, ind) == I_n`` (inverse tensor equation),
where ``I_n`` is the n-dimensional identity tensor and ``n`` is equal to ``input.ndim``.
The resulting tensor ``input_inv`` has shape equal to ``input.shape[ind:] + input.shape[:ind]``.

Supports input of ``float``, ``double``, ``cfloat`` and ``cdouble`` data types.

.. note:: If :attr:`input` is not invertible or does not satisfy the requirement
          ``prod(input.shape[ind:]) == prod(input.shape[:ind])``,
          then a RuntimeError will be thrown.

.. note:: When :attr:`input` is a 2-dimensional tensor and ``ind=1``, this function computes the
          (multiplicative) inverse of :attr:`input`, equivalent to calling :func:`torch.inverse`.

Args:
    input (Tensor): A tensor to invert. Its shape must satisfy ``prod(input.shape[:ind]) == prod(input.shape[ind:])``.
    ind (int): A positive integer that describes the inverse tensor equation. See :func:`torch.tensordot` for details.
               Default: 2.

Keyword args:
    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

Examples::

    >>> a = torch.eye(4 * 6).reshape((4, 6, 8, 3))
    >>> ainv = torch.linalg.tensorinv(a, ind=2)
    >>> ainv.shape
    torch.Size([8, 3, 4, 6])
    >>> b = torch.randn(4, 6)
    >>> torch.allclose(torch.tensordot(ainv, b), torch.linalg.tensorsolve(a, b))
    True

    >>> a = torch.randn(4, 4)
    >>> a_tensorinv = torch.linalg.tensorinv(a, ind=1)
    >>> a_inv = torch.inverse(a)
    >>> torch.allclose(a_tensorinv, a_inv)
    True
""")

tensorsolve = _add_docstr(_linalg.linalg_tensorsolve, r"""
linalg.tensorsolve(input, other, dims=None, *, out=None) -> Tensor

Computes a tensor ``x`` such that ``tensordot(input, x, dims=x.ndim) = other``.
The resulting tensor ``x`` has the same shape as ``input[other.ndim:]``.

Supports real-valued and complex-valued inputs.

.. note:: If :attr:`input` does not satisfy the requirement
          ``prod(input.shape[other.ndim:]) == prod(input.shape[:other.ndim])``
          after (optionally) moving the dimensions using :attr:`dims`, then a RuntimeError will be thrown.

Args:
    input (Tensor): "left-hand-side" tensor, it must satisfy the requirement
                    ``prod(input.shape[other.ndim:]) == prod(input.shape[:other.ndim])``.
    other (Tensor): "right-hand-side" tensor of shape ``input.shape[other.ndim]``.
    dims (Tuple[int]): dimensions of :attr:`input` to be moved before the computation.
                       Equivalent to calling ``input = movedim(input, dims, range(len(dims) - input.ndim, 0))``.
                       If None (default), no dimensions are moved.

Keyword args:
    out (Tensor, optional): The output tensor. Ignored if ``None``. Default: ``None``

Examples::

    >>> a = torch.eye(2 * 3 * 4).reshape((2 * 3, 4, 2, 3, 4))
    >>> b = torch.randn(2 * 3, 4)
    >>> x = torch.linalg.tensorsolve(a, b)
    >>> x.shape
    torch.Size([2, 3, 4])
    >>> torch.allclose(torch.tensordot(a, x, dims=x.ndim), b)
    True

    >>> a = torch.randn(6, 4, 4, 3, 2)
    >>> b = torch.randn(4, 3, 2)
    >>> x = torch.linalg.tensorsolve(a, b, dims=(0, 2))
    >>> x.shape
    torch.Size([6, 4])
    >>> a = a.permute(1, 3, 4, 0, 2)
    >>> a.shape[b.ndim:]
    torch.Size([6, 4])
    >>> torch.allclose(torch.tensordot(a, x, dims=x.ndim), b, atol=1e-6)
    True
""")
