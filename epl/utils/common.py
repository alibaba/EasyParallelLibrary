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
"""Common functions."""

from distutils.version import LooseVersion as Version
import re
from tensorflow.python.framework import device
from tensorflow.python.framework import ops
from tensorflow.python.framework.function import _FuncGraph
from tensorflow.python.framework.ops import Graph as TFGraph
from tensorflow.python.framework.versions import __version__
from epl.utils import constant


def is_func_graph(graph):
  """Check graph is func graph."""
  if isinstance(graph, _FuncGraph):
    return True
  if Version(__version__) >= Version("1.15"):
    from tensorflow.python.framework.func_graph import FuncGraph # pylint: disable=ungrouped-imports
    return isinstance(graph, FuncGraph)
  return False


def get_default_tf_graph():
  tf_graph = ops.get_default_graph()
  while is_func_graph(tf_graph):
    if hasattr(tf_graph, 'outer_graph'):
      tf_graph = getattr(tf_graph, 'outer_graph')
    else:
      tf_graph = getattr(tf_graph, '_outer_graph')
  if tf_graph.__class__ != TFGraph:
    raise RuntimeError("Default TF graph should be {}, but got {}".format(TFGraph, type(tf_graph)))
  return tf_graph


def is_indexed_slices(tensor):
  """Return whether tensor is instance of IndexedSlices."""
  return isinstance(tensor, ops.IndexedSlices)


def is_variable(op):
  """Is variable op."""
  return op.type in ["VariableV2", "Variable"]

def in_while_loop(op):
  """Is op in while loop."""
  from epl.ir.graph import Graph

  if hasattr(op, "_control_flow_context"):
    context = getattr(op, "_control_flow_context")
    if context and hasattr(context, "GetWhileContext"):
      return context.GetWhileContext() is not None
  return Graph.get().op2group.get(op.name, "").startswith("while_loop")

def is_const(op):
  """Is const op."""
  return op.type in ["Const", "ConstV2"]


def has_const_inputs_only(op):
  """Whether op has const inputs only or not."""
  op_inp = list(op.inputs)
  ret = all(is_const(ele.producer) for ele in op_inp)
  return ret



def get_device_string(job=constant.DEFAULT_TASK_NAME,
                      replica=0,
                      task=0,
                      device_type=constant.DEFAULT_DEVICE,
                      device_index=0):
  """Get device string using DeviceSpec."""
  return device.DeviceSpec(job=job,
                           replica=replica,
                           task=task,
                           device_type=device_type,
                           device_index=device_index).to_string()


def get_device_type(device_str):
  """Get device type from device str."""
  device_spec = device.DeviceSpec()
  device_spec.parse_from_string(device_str)
  return device_spec.device_type


def get_task_index_from_device_str(device_str):
  """Get task index from device str."""
  device_spec = device.DeviceSpec()
  device_spec.parse_from_string(device_str)
  return device_spec.task


def get_replica_prefix(replica_idx):
  """Get replica prefix."""
  return constant.REPLICA_PREFIX_FORMAT.format(replica_idx) if replica_idx else ""


def get_micro_batch_prefix(micro_batch_idx):
  """Get micro batch prefix."""
  return constant.MICRO_BATCH_PREFIX_FORMAT.format(micro_batch_idx) \
      if micro_batch_idx else ""


def get_replica_prefix_from_node_name(node_name):
  """Get replica prefix from operation name."""
  regex = re.compile(constant.REPLICA_PREFIX_FORMAT.format("([0-9]+)"))
  res = regex.search(node_name)
  return "" if res is None else res.group()


def get_micro_batch_prefix_from_node_name(node_name):
  """Get micro batch prefix from operation name."""
  regex = re.compile(constant.MICRO_BATCH_PREFIX_FORMAT.format("([0-9]+)"))
  res = regex.search(node_name)
  return "" if res is None else res.group()


def get_replica_index_from_node_name(node_name):
  """Get replica index from operation name."""
  res = re.search(constant.REPLICA_PREFIX_FORMAT.format("([0-9]+)"), node_name, re.I)
  return 0 if res is None else int(res.group(1))


def get_micro_batch_index_from_node_name(node_name):
  """Get micro batch index from operation name."""
  res = re.search(constant.MICRO_BATCH_PREFIX_FORMAT.format("([0-9]+)"), node_name,
                  re.I)
  return 0 if res is None else int(res.group(1))


def get_original_name_from_cloned_object(op_or_tensor_name):
  """Get name of original operation or tensor from cloned object."""
  replica_prefix = \
      get_replica_prefix_from_node_name(op_or_tensor_name)
  micro_batch_prefix = \
      get_micro_batch_prefix_from_node_name(op_or_tensor_name)
  original_name_offset = len(replica_prefix) + len(micro_batch_prefix)
  return op_or_tensor_name[original_name_offset:]


def update_tuple(origin_tuple, update_value, update_index):
  """Update tuple/namedtuple for specified update_index and update_value."""
  # Namedtuple is inherit from tuple.
  if not isinstance(origin_tuple, tuple):
    raise ValueError("Only tuple/namedtuple supported. Origin_tuple type: "
                     "%s." % type(origin_tuple))

  if update_index >= len(origin_tuple):
    raise ValueError("Update index is out of range. Length of original tuple "
                     "%s, Update index: %s." %
                     (len(origin_tuple), update_index))

  values = []
  for index, item in enumerate(origin_tuple):
    if index == update_index:
      values.append(update_value)
    else:
      values.append(item)

  def _is_namedtuple(x):
    base = type(x).__bases__
    if len(base) == 1 and base[0] == tuple:
      return True
    return False

  if _is_namedtuple(origin_tuple):
    return type(origin_tuple)(*values)
  return tuple(values)
