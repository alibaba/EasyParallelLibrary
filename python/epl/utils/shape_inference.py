# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Inference TF Graph Shape"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.framework.common_shapes import call_cpp_shape_fn
from tensorflow.python.framework import tensor_util

from epl.ir.graph import Graph
from epl.ir.phase import ModelPhase


def get_cpp_shape(op):
  """Call call_cpp_shape_fn to infer op shapes"""
  # avoid error: AttributeError: 'Tensor' object has no attribute '_handle_data'
  shape_list = [get_tf_tensor_shape(t) for t in op.outputs]
  if all(s is not None for s in shape_list):
    return shape_list
  for t in op.inputs:
    if not hasattr(t, "_handle_data"):
      t._handle_data = None  # pylint: disable=protected-access
  shapes = call_cpp_shape_fn(op)
  shape_list = []
  for shape in shapes["shapes"]:
    if shape.__class__.__name__ == "TensorShape":
      shape_list.append(shape.as_list())
    else:
      shape_list.append([int(s.size) for s in shape.dim])
  return shape_list


def try_get_shape_value(tensor):
  """try to get the tensor shape value"""
  op = tensor.op
  if op.type == "Pack":
    return get_pack_val(op)
  if op.type == "StridedSlice":
    return get_stride_slice_value(op)
  if op.type in ["Const", "ConstV2"]:
    return get_const_value(op)
  if op.type == "Mul":
    return get_mul_value(op)
  if op.type == "Prod":
    return get_prod_value(op)
  if op.type in ["Gather", "GatherV2"]:
    return get_gather_value(op)
  if op.type in ["Shape"]:
    return get_cpp_shape(op.inputs[0].op)[0]
  if op.type in ["Reshape"]:
    return try_get_shape_value(op.inputs[1])
  if op.type in ["Fill"]:
    return try_get_shape_value(op.inputs[0])
  if op.type in ["Concat", "ConcatV2"]:
    return get_concat_value(op)
  return None


def check_type(op, types):
  """Check type for op"""
  assert op.type in types, \
      "Op type {} not valid, should be {}.".format(op.type, types)


def get_concat_value(op):
  """Get value for Concat/ConcatV2"""
  check_type(op, ["Concat", "ConcatV2"])
  axis = try_get_shape_value(op.inputs[-1])[0]
  values = [try_get_shape_value(i) for i in op.inputs[:-1]]
  return np.concatenate(values, axis=axis)


def get_pack_val(op):
  """Get value for Pack"""
  check_type(op, ["Pack"])
  axis = int(op.node_def.attr['axis'].i)
  assert axis == 0, "Current only process pack axis 0, got {}".format(axis)
  input_vals = [try_get_shape_value(i) for i in op.inputs]
  input_vals = [i[0] if isinstance(i, list) else i for i in input_vals]
  return input_vals


def get_tf_tensor_shape(tensor):
  """Get tensor shape, if there is unkown tensor, set it as None"""
  shape = []
  try:
    shape = tensor.get_shape().as_list()
    if any(s is None for s in shape):
      return None
    return shape
  except Exception:  # pylint: disable=broad-except
    shape = None
  return shape


def get_mul_value(op):
  """Get value for Mul"""
  check_type(op, ["Mul"])
  x = try_get_shape_value(op.inputs[0])
  y = try_get_shape_value(op.inputs[1])
  return np.multiply(x, y)


def get_prod_value(op):
  """Get value for Prod"""
  check_type(op, ["Prod"])
  x = try_get_shape_value(op.inputs[0])
  axis = try_get_shape_value(op.inputs[1])[0]
  assert axis == 0, "Current only process Prod axis 0, got {}".format(axis)
  return np.prod(x)


def get_gather_value(op):
  """Get value for Gather"""
  check_type(op, ["Gather", "GatherV2"])
  params = try_get_shape_value(op.inputs[0])
  slices = try_get_shape_value(op.inputs[1])
  return [params[i] for i in slices]


def get_const_value(op):
  """Get value for Shape"""
  check_type(op, ["Const", "ConstV2"])
  node_def_tensor = op.node_def.attr["value"].tensor
  np_data = tensor_util.MakeNdarray(node_def_tensor)
  data = np_data.tolist()
  if not isinstance(data, list):
    data = [data]
  return data


def get_stride_slice_value(op):
  """Get value for StridedSlice"""
  check_type(op, ["StridedSlice"])
  inputs = [try_get_shape_value(i) for i in op.inputs]
  data = inputs[0]
  begin = inputs[1][0]
  end = inputs[2][0]
  strides = inputs[3][0]
  return data[begin:end:strides]


def get_batch_size(ops):
  """Get batch size of model."""
  for op in ops:
    if op.type in ["Const", "ConstV2"] and op.name == "batch_size":
      return get_const_value(op)[0]
  return -1


def set_input_batch_size(ops, batch_size):
  """Set batch size for input data."""
  for op in ops:
    if op.type == "IteratorGetNext":
      for op_out in op.outputs:
        shape = op_out.shape.as_list()
        if shape[0] is None or shape[0] == -1:
          shape[0] = batch_size
          op_out.set_shape(shape)
      return


def is_gradient_op(op):
  """Check if op is gradient op."""
  epl_op = Graph.get().get_operation_by_name(op.name)
  return epl_op is not None and epl_op.phase == ModelPhase.BACKWARD


def filter_ops(ops):
  """Filter gradients ops."""
  # TODO(sayang): tackle gradient op shape.
  not_gradients = lambda op: not is_gradient_op(op)
  return list(filter(not_gradients, ops))


def infer_shape(tf_graph):
  """Infer shape for Tensorflow ops."""
  ops = tf_graph.get_operations()
  ops = filter_ops(ops)
  batch_size = get_batch_size(ops)
  try:
    set_input_batch_size(ops, batch_size)
  except Exception as e:  # pylint: disable=broad-except
    tf.logging.warn("shape inference fail for inputs: {}".format(e))
  for op in ops:
    infer_shape_for_op(op)
  return tf_graph


def has_unknown_shape(shape):
  """Check if shape list has unknown dimension."""
  return any(s is None or s < 0 for s in shape)


def infer_shape_for_op(op):
  """Infer shape for an op."""
  if not any(get_tf_tensor_shape(out) is None for out in op.outputs):
    return
  try:
    shape_list = get_cpp_shape(op)
    for tensor, shape in zip(op.outputs, shape_list):
      if not has_unknown_shape(shape):
        tensor.set_shape(shape)
        continue
      shape = try_get_shape_value(tensor)
      if shape is not None:
        tensor.set_shape(shape)
  except Exception as e:  # pylint: disable=broad-except
    tf.logging.warn("shape inference fail for op {}: {}".format(op.name, e))
