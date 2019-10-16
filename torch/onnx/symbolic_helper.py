from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from torch._C import ListType
import warnings
from sys import maxsize as maxsize

import torch.onnx
# This import monkey-patches graph manipulation methods on Graph, used for the
# ONNX symbolics
import torch.onnx.utils

from functools import wraps


# Note [Edit Symbolic Files]
# EDITING THIS FILE AND SYMBOLIC_OPSET<VERSION> FILES? READ THIS FIRST!
#
# - These files is ONLY for ATen operators (e.g., operators that show up in the
#   trace as aten::blah).  If you need to special case a primitive operator,
#   look at _run_symbolic_function
# - Parameter ordering does NOT necessarily match what is in VariableType.cpp;
#   tensors are always first, then non-tensor arguments.
# - Parameter names must *exactly* match the names in VariableType.cpp, because
#   dispatch is done with keyword arguments.
# - Looking for inplace ops?  They're detected by the trailing underscore, and
#   transparently dispatched to their non inplace versions in
#   'run_symbolic_function'.   See Note [Export inplace]
#
# ----------------------------------------------------------------------------------
# A note on Tensor types
# ----------------------------------------------------------------------------------
#
# In general, we should avoid depending on the type of Tensor Values contained
# within the trace graph. However, this is sometimes unavoidable (due to ONNX
# spec requirements, etc). The TensorType object has accessors for these properties
# that return the property if it is statically known and return nullopt otherwise. 
#
# In general, we should prefer to rely on the least specific information possible.
# For example, not relying on tensor properties at all is better than relying
# on the number of dimensions which is better than relying on
# concrete shapes. Doing so will make the export symbolics
# more robust to different graphs.

# ---------------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------------

# Save some builtins as locals, because we'll shadown them below
_sum = sum


def _parse_arg(value, desc):
    if desc == 'none':
        return value
    if desc == 'v' or not _is_value(value):
        return value
    if value.node().mustBeNone():
        return None
    if value.node().kind() == 'onnx::Constant':
        tval = value.node()['value']
        if desc == 'i':
            return int(tval)
        elif desc == 'f':
            return float(tval)
        elif desc == 'b':
            return bool(tval)
        elif desc == 's':
            return str(tval)
        elif desc == 't':
            return tval
        elif desc == 'is':
            return [int(v) for v in tval]
        else:
            raise RuntimeError("ONNX symbolic doesn't know to interpret Constant node")
    elif value.node().kind() == 'prim::ListConstruct':
        if desc == 'is':
            for v in value.node().inputs():
                if v.node().kind() != 'onnx::Constant':
                    raise RuntimeError("Failed to export an ONNX attribute, "
                                       "since it's not constant, please try to make "
                                       "things (e.g., kernel size) static if possible")
            return [int(v.node()['value']) for v in value.node().inputs()]
        else:
            raise RuntimeError("ONNX symbolic doesn't know to interpret ListConstruct node")

    raise RuntimeError("Unexpected node type: {}".format(value.node().kind()))


def _maybe_get_const(value, desc):
    if _is_value(value) and value.node().kind() == 'onnx::Constant':
        return _parse_arg(value, desc)
    return value


def _maybe_get_scalar(value):
    value_t = _maybe_get_const(value, 't')
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value


def _get_const(value, desc, arg_name):
    if _is_value(value) and value.node().kind() != 'onnx::Constant':
        raise RuntimeError("ONNX symbolic expected a constant value of the {} argument, got `{}`".format(arg_name, value))
    return _parse_arg(value, desc)


def _unpack_list(list_value):
    list_node = list_value.node()
    assert list_node.kind() == "prim::ListConstruct"
    return list(list_node.inputs())


# Check if list_value is output from prim::ListConstruct
# This is usually called before _unpack_list to ensure the list can be unpacked.
def _is_packed_list(list_value):
    return _is_value(list_value) and list_value.node().kind() == "prim::ListConstruct"


def parse_args(*arg_descriptors):
    def decorator(fn):
        fn._arg_descriptors = arg_descriptors

        def wrapper(g, *args):
            # some args may be optional, so the length may be smaller
            assert len(arg_descriptors) >= len(args)
            args = [_parse_arg(arg, arg_desc) for arg, arg_desc in zip(args, arg_descriptors)]
            return fn(g, *args)
        # In Python 2 functools.wraps chokes on partially applied functions, so we need this as a workaround
        try:
            wrapper = wraps(fn)(wrapper)
        except Exception:
            pass
        return wrapper
    return decorator


def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x.item()


def _if_scalar_type_as(g, self, tensor):
    """
    Convert self into the same type of tensor, as necessary.

    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    if isinstance(self, torch._C.Value):
        return self

    scalar_type = tensor.type().scalarType()
    if scalar_type:
        ty = scalar_type.lower()
        return getattr(self, ty)()

    return self


def _is_none(x):
    return x.node().mustBeNone()

def _is_value(x):
    return isinstance(x, torch._C.Value)


def _is_tensor_list(x):
    return x.type().isSubtypeOf(ListType.ofTensors())


def _unimplemented(op, msg):
    warnings.warn("ONNX export failed on " + op + " because " + msg + " not supported")


def _black_list_in_opset(name):
    def symbolic_fn(*args, **kwargs):
        raise RuntimeError("ONNX export failed on {}, which is not implemented for opset {}. "
                           "Try exporting with other opset versions."
                           .format(name, _export_onnx_opset_version))
    return symbolic_fn


def _try_get_scalar_type(*args):
    for arg in args:
        try:
            return arg.type().scalarType()
        except RuntimeError:
            pass
    return None


def _slice_helper(g, input, axes, starts, ends, steps=None, dynamic_slice=False):
    if _export_onnx_opset_version <= 9:
        from torch.onnx.symbolic_opset9 import _slice
        return _slice(g, input, axes, starts, ends)
    else:
        from torch.onnx.symbolic_opset10 import _slice
        return _slice(g, input, axes, starts, ends, steps, dynamic_slice)


def _is_fp(value):
    if value:
        type = value.type().scalarType()
        return (type == 'Float') or (type == 'Double') or (type == 'Half')
    return False


def _sort_helper(g, input, dim, decending=True, out=None):
    if out is not None:
        _unimplemented("Sort", "Out parameter is not supported")
    shape_ = g.op("Shape", input)
    axis = g.op("Constant", value_t=torch.tensor(0, dtype=torch.int64))
    start = g.op("Constant", value_t=torch.tensor(dim, dtype=torch.int64))
    end = g.op("Constant", value_t=torch.tensor(dim + 1, dtype=torch.int64))
    slice_ = _slice_helper(g, shape_, axes=axis, starts=start, ends=end, steps=None, dynamic_slice=True)
    if _export_onnx_opset_version <= 10:
        if not decending:
            _unimplemented("Sort", "Ascending is not supported")
        return g.op("TopK", input, slice_, axis_i=dim, outputs=2)
    else:
        return g.op("TopK", input, slice_, axis_i=dim, largest_i=decending, outputs=2)


def _topk_helper(g, input, k, dim, largest=True, sorted=False, out=None):
    if out is not None:
        _unimplemented("TopK", "Out parameter is not supported")
    if _export_onnx_opset_version <= 10:
        if not largest:
            _unimplemented("TopK", "Ascending is not supported")
            return g.op("TopK", input, k_i=k, axis_i=dim, outputs=2)
        k = _maybe_get_const(k, 'i')
        if not _is_value(k):
            k = g.op("Constant", value_t=torch.tensor(k, dtype=torch.int64))
        from torch.onnx.symbolic_opset9 import unsqueeze
        k = unsqueeze(g, k, 0)
        return g.op("TopK", input, k, axis_i=dim, outputs=2)
    else:
        return g.op("TopK", input, k, axis_i=dim, largest_i=largest, sorted_i=sorted, outputs=2)


def _interpolate_warning(interpolate_mode):
    onnx_op = "onnx:Resize" if _export_onnx_opset_version >= 10 else "onnx:Upsample"
    warnings.warn("You are trying to export the model with " + onnx_op + " for ONNX opset version "
                  "" + str(_export_onnx_opset_version) + ". "
                  "This operator might cause results to not match the expected results by PyTorch.\n"
                  "ONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. "
                  "Attributes to determine how to transform the input were added in onnx:Resize in opset 11 "
                  "to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).\n"
                  "We recommend using opset 11 and above for models using this operator. ")


def _interpolate_size_to_scales(g, input, output_size, dim):
    output_size = _maybe_get_const(output_size, 'is')
    if _is_value(output_size):
        offset = 2
        offsets = g.op("Constant", value_t=torch.ones(offset, dtype=torch.float32))
        dividend = g.op("Cast", output_size, to_i=cast_pytorch_to_onnx["Float"])
        divisor = _slice_helper(g, g.op("Shape", input), axes=[0], ends=[maxsize], starts=[offset])
        divisor = g.op("Cast", divisor, to_i=cast_pytorch_to_onnx["Float"])
        scale_dims = g.op("Div", dividend, divisor)
        scales = g.op("Concat", offsets, scale_dims, axis_i=0)
    else:
        scales_constant = [1. if i < 2 else
                           float(output_size[-(dim - i)]) / float(input.type().sizes()[-(dim - i)])
                           for i in range(0, dim)]
        scales = g.op("Constant", value_t=torch.tensor(scales_constant, dtype=torch.float32))
    return scales


def _interpolate_get_scales(g, scale_factor, dim):
    from torch.onnx.symbolic_opset9 import unsqueeze

    offsets = g.op("Constant", value_t=torch.ones(2, dtype=torch.float32))
    if _is_packed_list(scale_factor):
        scale_factor = _unpack_list(scale_factor)
        scales = []
        for dim_scale_factor in scale_factor:
            dim_scale_factor = unsqueeze(g, dim_scale_factor, 0)
            dim_scale_factor = g.op("Cast", dim_scale_factor, to_i=cast_pytorch_to_onnx["Float"])
            scales.append(dim_scale_factor)
    else:
        scale_factor = unsqueeze(g, scale_factor, 0)
        scale_factor = g.op("Cast", scale_factor, to_i=cast_pytorch_to_onnx["Float"])
        scales = [scale_factor for i in range(dim - 2)]
    scale_factor = g.op("Concat", offsets, *scales, axis_i=0)
    return scale_factor


def _interpolate_get_scales_and_mode(g, input, size, scale_factor, mode , align_corners):
    from torch.onnx.symbolic_opset9 import unsqueeze
    mode = _maybe_get_const(mode, 's')
    _interpolate_warning(mode)

    align_corners = _maybe_get_const(align_corners, 'b')
    if not _is_none(align_corners) and align_corners:
        return _unimplemented("interpolate", "align_corners == True")

    dim = input.type().dim()

    if not _is_none(scale_factor):
        scale_factor = _interpolate_get_scales(g, scale_factor, dim)
    elif not _is_none(size):
        if not _is_packed_list(size):
            is_scalar = ((_maybe_get_const(size, 't').dim() == 0))
            if is_scalar:
                size = unsqueeze(g, size, 0)
                size = [size for i in range(dim - 2)]
                size = g.op("Concat", *size, axis_i=0)
        scale_factor = _interpolate_size_to_scales(g, input, size, dim)
    else:
        _unimplemented("Both size and scales are None in __interpolate")
    return scale_factor, mode


def _scatter_helper(g, self, dim, index, src):
    if _export_onnx_opset_version <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        from torch.onnx.symbolic_opset11 import scatter
    return scatter(g, self, dim, index, src)


def _arange_cast_helper(g, end, start=None, step=None, dtype=None):
    # This logic is based on torch.arange docs. If 'dtype' is provided,
    # infer input types from dtype. If not, then check if any of start, stop,
    # or step are floating point, and infer the type from get_default.
    # Otherwise, the dtype is inferred to be torch.int64.
    if dtype:
        type = dtype
    else:
        type = scalar_type_to_pytorch_type.index(torch.get_default_dtype())\
            if _is_fp(start) or _is_fp(end) or _is_fp(step) else 4  # default torch.int64

    start = g.op("Cast", start, to_i=scalar_type_to_onnx[type]) if start else None
    end = g.op("Cast", end, to_i=scalar_type_to_onnx[type]) if end else None
    step = g.op("Cast", step, to_i=scalar_type_to_onnx[type]) if step else None
    return type, end, start, step


def _size_helper(g, self, dim):
    full_shape = g.op("Shape", self)
    from torch.onnx.symbolic_opset9 import select
    return select(g, full_shape, g.op("Constant", value_t=torch.tensor([0])), dim)


def _index_fill_reshape_helper(g, self, dim, index):
    # 1. reshape index => [1, ..., 1, dim, 1, ..., 1]
    # 2. expand index => [..., dim, ...], same shape as self except for dim.
    # 3. expand value as well.
    # 4. apply onnx::scatter.

    from torch.onnx.symbolic_opset9 import expand
    if _export_onnx_opset_version <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        from torch.onnx.symbolic_opset11 import scatter

    if self.type().dim() is None:
        return _unimplemented("index_fill", "input rank not accesible")
    self_dim = self.type().dim()
    dim_value = _parse_arg(dim, 'i')
    unsqueezed_index = g.op("Unsqueeze", index, axes_i=[i for i in range(self_dim) if i != dim_value])
    expanded_index_shape = scatter(g, g.op("Shape", self), 0,
                                   g.op("Unsqueeze", dim, axes_i=[0]), g.op("Shape", index))
    expanded_index = expand(g, unsqueezed_index, expanded_index_shape, None)
    return expanded_index_shape, expanded_index

# ---------------------------------------------------------------------
# ONNX operator version
# ---------------------------------------------------------------------

# READ ME BEFORE EDITING _default_onnx_opset_version:
#
# The variable below controls which ONNX operator set version we are
# targeting. THIS VARIABLE HAS SEMANTIC EFFECT! Say a breaking
# change occurred in version 8. As long as this variable < 8, you can
# export models targeting the old behavior. However, if you bump
# this variable to 8 or later, the breaking change will take into effect:
# you MUST adjust any symbolic affected by breaking changes. The ONNX
# spec publishes a *comprehensive* list of BC-breaking changes for every
# operator revision at:
#
#   https://github.com/onnx/onnx/blob/master/docs/Changelog.md
#
# Please be sure to go through and check all of our implementations here before
# increasing this number. This includes symbolic definitions NOT in this
# file, so grep for "OpName" (with quotes)
#
# Besides, opset_version can be specified in the invocation of export()
# and export_to_pretty_string(), and _export_onnx_opset_version will be set
# and the symbolic functions should check it to determine the behavior
# of the exporter.


_default_onnx_opset_version = 9
_onnx_master_opset = 10
_onnx_stable_opsets = [7, 8, 9, 10, 11]
_export_onnx_opset_version = _default_onnx_opset_version


def _set_opset_version(opset_version):
    global _export_onnx_opset_version
    if opset_version == _default_onnx_opset_version:
        _export_onnx_opset_version = opset_version
        return
    if opset_version in _onnx_stable_opsets + [_onnx_master_opset]:
        _export_onnx_opset_version = opset_version
        return
    raise ValueError("Unsupported ONNX opset version: " + str(opset_version))

_operator_export_type = None
def _set_operator_export_type(operator_export_type):
    global _operator_export_type
    _operator_export_type = operator_export_type

# Metaprogram symbolics for each ATen native specialized cast operator.
# For e.g. we specify a function named `_cast_uint8_t` that instantiates an
# ONNX cast node with `to` attribute 'UINT8'
#
# TODO: remove these once we support Type's in the JIT IR and we can once again
# use the unified toType operator
cast_pytorch_to_onnx = {
    'Byte': torch.onnx.TensorProtoDataType.UINT8,
    'Char': torch.onnx.TensorProtoDataType.INT8,
    'Double': torch.onnx.TensorProtoDataType.DOUBLE,
    'Float': torch.onnx.TensorProtoDataType.FLOAT,
    'Half': torch.onnx.TensorProtoDataType.FLOAT16,
    'Int': torch.onnx.TensorProtoDataType.INT32,
    'Long': torch.onnx.TensorProtoDataType.INT64,
    'Short': torch.onnx.TensorProtoDataType.INT16,
    'Bool': torch.onnx.TensorProtoDataType.BOOL,
    'ComplexFloat': torch.onnx.TensorProtoDataType.COMPLEX64,
    'ComplexDouble': torch.onnx.TensorProtoDataType.COMPLEX128,
    'Undefined': torch.onnx.TensorProtoDataType.UNDEFINED,
}

scalar_name_to_pytorch = {
    'uint8_t': 'Byte',
    'int8_t': 'Char',
    'double': 'Double',
    'float': 'Float',
    'half': 'Half',
    'int': 'Int',
    'int64_t': 'Long',
    'int16_t': 'Short',
    'bool': 'Bool',
    'complex64': '',
    'complex128': ''
}


# This indicates each scalar type's corresponding
# torch type. Related source:
# https://github.com/pytorch/pytorch/blob/da7468853ae322252270bbb58032668bd21b7457/c10/core/ScalarType.h
scalar_type_to_pytorch_type = [
    torch.uint8,        # 0
    torch.int8,         # 1
    torch.short,        # 2
    torch.int,          # 3
    torch.int64,        # 4
    torch.half,         # 5
    torch.float,        # 6
    torch.double,       # 7
    torch.complex64,    # 9
    torch.complex128,   # 10
    torch.bool,         # 11
]


def _cast_func_template(to_i, g, input, non_blocking):
    return g.op("Cast", input, to_i=to_i)


scalar_type_to_onnx = [
    cast_pytorch_to_onnx["Byte"],
    cast_pytorch_to_onnx["Char"],
    cast_pytorch_to_onnx["Short"],
    cast_pytorch_to_onnx["Int"],
    cast_pytorch_to_onnx["Long"],
    cast_pytorch_to_onnx["Half"],
    cast_pytorch_to_onnx["Float"],
    cast_pytorch_to_onnx["Double"],
    cast_pytorch_to_onnx["Undefined"],
    cast_pytorch_to_onnx["ComplexFloat"],
    cast_pytorch_to_onnx["ComplexDouble"],
    cast_pytorch_to_onnx["Bool"],
]
