import torch
from torch.nn.modules.utils import _single, _pair, _triple
import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _unimplemented, _black_list_in_opset
import torch.onnx.symbolic_opset9


# EDITING THIS FILE? READ THIS FIRST!
# see Note [Edit Symbolic Files] in symbolic_helper.py

# This file exports ONNX ops for opset 10
# Opset 10 is supported by ONNX release 1.5.0
# release on 04/24/19


# Blacklist operators for this opset version.
# These operators have been updated in ONNX but not re-implemented here.
# It is very important to blacklist these operators to avoid exporting
# models with mixed versions of operators.
# TODO : add support for the blacklisted operators in black_listed_operators
black_listed_operators = ["upsample_nearest2d", "upsample_bilinear2d"]

for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)


# Add new operator here
@parse_args('v', 'i', 'i', 'i', 'i')
def topk(g, self, k, dim, largest, sorted, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported for topk")
    if not largest:
        _unimplemented("TopK", "Ascending TopK is not supported")
    k = g.op("Constant", value_t=torch.tensor(k, dtype=torch.int64))
    from torch.onnx.symbolic_opset9 import unsqueeze
    k = unsqueeze(g, k, 0)
    return g.op("TopK", self, k, axis_i=dim, outputs=2)


def _max_pool(name, tuple_fn, ndims, return_indices):
    @parse_args('v', 'is', 'is', 'is', 'is', 'i')
    def symbolic_fn(g, input, kernel_size, stride, padding, dilation, ceil_mode):
        if not stride:
            stride = kernel_size
        kwargs = {
            'kernel_shape_i': tuple_fn(kernel_size),
            'pads_i': tuple_fn(padding) * 2,
            'strides_i': tuple_fn(stride),
            'ceil_mode_i': ceil_mode,
        }
        if set(tuple_fn(dilation)) != {1}:
            kwargs['dilations_i'] = tuple_fn(dilation)
        # easy but hacky way to get flattened indices values
        # to be used to convert the indices values to non-flattened.
        # In ONNX the indices are computed as a flatten 1-D tensor,
        # so the values in indices are in [0, N x C x D1 x ... x Dn).
        # To convert the indices to the same format used by Pytorch,
        # we first execute a maxpool with a kernel and stride of 1 on the same input.
        # This will result in a tensor of indices in which each index will have it's own value.
        # Using this tensor as a reference, we extract the first index of each axis and substract
        # it from each index of this axis in the indices to convert.
        # This step will result in a tensor were each dimension has values of indices within
        # the dimension it is in.
        # For more information :
        # https://github.com/pytorch/pytorch/pull/16455#issuecomment-460776407
        if return_indices:
            r, indices = g.op("MaxPool", input, outputs=2, **kwargs)
            _, flattened_indices = g.op("MaxPool", input, outputs=2,
                                        kernel_shape_i=[1 for _ in range(ndims)],
                                        strides_i=[1 for _ in range(ndims)])
            # convert indices to have non-flattened indices values
            s = _slice_op(g, flattened_indices, axes=[2 + i for i in range(ndims)],
                          starts=tuple_fn(0), ends=tuple_fn(1))
            indices = sub(g, indices, s)
            return r, indices
        else:
            r = g.op("MaxPool", input, outputs=1, **kwargs)
            return r

    return symbolic_fn


max_pool1d = _max_pool("max_pool1d", _single, 1, return_indices=False)
max_pool2d = _max_pool("max_pool2d", _pair, 2, return_indices=False)
max_pool3d = _max_pool("max_pool3d", _triple, 3, return_indices=False)
max_pool1d_with_indices = _max_pool("max_pool1d_with_indices", _single, 1, return_indices=True)
max_pool2d_with_indices = _max_pool("max_pool2d_with_indices", _pair, 2, return_indices=True)
max_pool3d_with_indices = _max_pool("max_pool3d_with_indices", _triple, 3, return_indices=True)


def _avg_pool(name, tuple_fn):
    @parse_args('v', 'is', 'is', 'is', 'i', 'i')
    def symbolic_fn(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad):
        if not stride:
            stride = kernel_size
        padding = tuple(tuple_fn(padding))
        if count_include_pad:
            input = g.op("Pad", input,
                         pads_i=((0,) * 2 + padding) * 2,
                         mode_s='constant',
                         value_f=0.)
            padding = (0,) * len(padding)
        output = g.op("AveragePool", input,
                      kernel_shape_i=tuple_fn(kernel_size),
                      strides_i=tuple_fn(stride),
                      pads_i=padding * 2,
                      ceil_mode_i=ceil_mode)
        return output
    return symbolic_fn


avg_pool1d = _avg_pool('avg_pool1d', _single)
avg_pool2d = _avg_pool('avg_pool2d', _pair)
avg_pool3d = _avg_pool('avg_pool3d', _triple)


def slice_op(g, input, axes, starts, ends, steps=None, dynamic_slice=False):
    if dynamic_slice:
        starts = g.op("Unsqueeze", starts, axes_i=[0])
        ends = g.op("Unsqueeze", ends, axes_i=[0])
        axes = g.op("Unsqueeze", axes, axes_i=[0])
    else:
        assert len(starts) == len(ends)
        assert len(starts) == len(axes)
        assert steps is None or len(starts) == len(steps)
        if len(starts) == 1 and starts[0] == 0 and ends[0] == 9223372036854775807 \
           and (steps is None or (len(steps) == 1 and steps[0] == 1)):
            return input    
        axes = g.op("Constant", value_t=torch.tensor(axes))
        starts = g.op("Constant", value_t=torch.tensor(starts))
        ends = g.op("Constant", value_t=torch.tensor(ends))
    if steps is None:
        return g.op("Slice", input, starts, ends, axes)
    steps = g.op("Constant", value_t=torch.tensor(steps))
    return g.op("Slice", input, starts, ends, axes, steps)


@parse_args('v', 'v', 'v', 'v', 'i')
def slice(g, self, dim, start, end, step):
    if (start.node().kind() != 'onnx::Constant' or
       end.node().kind() != 'onnx::Constant' or dim.node().kind() != 'onnx::Constant'):
        dynamic_slice = True
    else:
        start = [sym_help._parse_arg(start, 'i')]
        end = [sym_help._parse_arg(end, 'i')]
        dim = [sym_help._parse_arg(dim, 'i')]
        dynamic_slice = False
    return sym_help._slice_op(g, self, axes=dim, starts=start, ends=end, steps=[step], dynamic_slice=dynamic_slice)


@parse_args('v', 'is')
def flip(g, input, dims):
    return sym_help._slice_op(g, input, axes=dims,
                              starts=[-1] * len(dims),
                              ends=[-9223372036854775807] * len(dims),
                              steps=[-1] * len(dims))
