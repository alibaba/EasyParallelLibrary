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
"""Operator functions for doing parallelism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from distutils.version import LooseVersion as Version
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.framework.function import _DefinedFunction
from tensorflow.python.framework.versions import __version__
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.util import compat

from epl.communicators.collective_communicator import CollectiveCommunicator
from epl.env import Env
from epl.ir.graph import Graph
from epl.ir.phase import ModelPhase
from epl.ir.operation import Operation as EplOperation
from epl.ir.tensor import Tensor as EplTensor
from epl.utils import constant, common


class Colocate(object):
  """Colocate operations."""

  def __init__(self, colocate_obj):
    if isinstance(colocate_obj, ops.Tensor):
      colocate_obj = Graph.get().get_tensor_by_name(colocate_obj.name).op
    elif isinstance(colocate_obj, ops.Operation):
      colocate_obj = Graph.get().get_operation_by_name(colocate_obj.name)
    if not isinstance(colocate_obj, (EplTensor, EplOperation)):
      raise RuntimeError("Colocate only supports tensor and operation as input, but got {}".format(colocate_obj))
    self._taskgraph = colocate_obj.taskgraph
    self._phase = colocate_obj.phase
    self._graph = self._taskgraph.graph
    self._device = colocate_obj.device

  def __enter__(self):
    self._graph.colocate_taskgraph = self._taskgraph
    self._graph.colocate_phase = self._phase
    self._graph.colocate_device = self._device

  def __exit__(self, unused_exception_type, unused_exception_value,
               unused_traceback):
    self._graph.colocate_taskgraph = None
    self._graph.colocate_phase = None
    self._graph.colocate_device = None


def node_clone_for_pipeline(graph, orig_op, micro_batch_idx, device):
  """Clone a operation to 'device' from 'orig_op' for pipeline."""
  micro_batch_prefix = common.get_micro_batch_prefix(micro_batch_idx)
  # get node def
  node_def = copy.deepcopy(orig_op.node_def)
  node_def.name = micro_batch_prefix + node_def.name
  frame_name = node_def.attr.get('frame_name')
  if frame_name:
    node_def.attr.get('frame_name').s = compat.as_bytes(micro_batch_prefix, constant.ENCODING) + frame_name.s

  op_def = copy.deepcopy(orig_op.op_def)
  output_types = orig_op.output_types[:]
  input_types = orig_op.input_types[:]

  graph.unready_inputs_cache[node_def.name] = dict()
  graph.unready_control_inputs_cache[node_def.name] = dict()

  # get inputs
  inputs = []
  for inp_idx, inp in enumerate(orig_op.inputs):
    if graph.is_dataset_type(orig_op) or \
        graph.is_dataset_related(inp.producer) or \
        graph.is_vars_related(inp.producer) or \
        graph.is_global_step_related(inp):
      name = inp.name
    else:
      name = micro_batch_prefix + inp.name
    if name in graph.tensors:
      inputs.append(graph.get_tensor_by_name(name).primitive_obj)
    else:
      tensor = graph.get_tensor_by_name(inp.name)
      if tensor.producer.get_control_flow_context() is not None:
        graph.original_context_cache[tensor.producer.name] = tensor.producer.get_control_flow_context()
        tensor.producer.set_control_flow_context(None)
      inputs.append(tensor.primitive_obj)
      graph.unready_inputs_cache[node_def.name][inp_idx] = name

  # get control inputs
  control_inputs = []
  old_control_inputs = list(orig_op.control_inputs)
  for c_inp in old_control_inputs:
    c_inp = c_inp if isinstance(c_inp, ops.Operation) else c_inp.producer
    c_inp = graph.get_operation_by_name(c_inp.name)
    if graph.is_dataset_related(orig_op) or \
        graph.is_dataset_related(c_inp) or \
        graph.is_vars_related(c_inp) or \
        graph.is_global_step_related(c_inp):
      name = c_inp.name
    else:
      name = micro_batch_prefix + c_inp.name
    if name in graph.operations:
      control_inputs.append(graph.get_operation_by_name(name).primitive_obj)
    else:
      op = graph.get_operation_by_name(c_inp.name)
      if op.get_control_flow_context() is not None:
        graph.original_context_cache[op.name] = op.get_control_flow_context()
        op.set_control_flow_context(None)
      control_inputs.append(op.primitive_obj)
      graph.unready_control_inputs_cache[node_def.name][op.name] = name

  if not graph.unready_inputs_cache[node_def.name]:
    del graph.unready_inputs_cache[node_def.name]
  if not graph.unready_control_inputs_cache[node_def.name]:
    del graph.unready_control_inputs_cache[node_def.name]

  with ModelPhase(orig_op.phase):
    graph.current_cloned_taskgraph = orig_op.taskgraph
    new_op = ops.Operation(node_def,
                           ops.get_default_graph(),
                           inputs,
                           output_types,
                           control_inputs,
                           input_types,
                           None,
                           op_def=op_def)
    new_op._set_device(device)  # pylint: disable=protected-access


def node_clone_for_replicas(graph, orig_op, replica_idx, device):
  """Clone a operation to 'device' from 'orig_op' for data parallelism."""
  replica_prefix = common.get_replica_prefix(replica_idx)
  # get node def
  node_def = copy.deepcopy(orig_op.node_def)
  loc_attr = node_def.attr.get('_class')
  node_def.name = replica_prefix + node_def.name
  if loc_attr:
    for idx in range(len(loc_attr.list.s)):
      loc_attr.list.s[idx] = loc_attr.list.s[idx].replace(b'@', b'@' + compat.as_bytes(replica_prefix, constant.ENCODING))
  for attr_name in ["frame_name", "shared_name"]:
    attr_value = node_def.attr.get(attr_name)
    if attr_value and attr_value.s:
      node_def.attr.get(attr_name).s = compat.as_bytes(replica_prefix, constant.ENCODING) + attr_value.s

  if graph.clone_dataset_related_ops and \
      graph.is_dataset_related(orig_op) and \
      node_def.attr.get('value') and \
      node_def.attr.get("value").tensor and \
      node_def.attr.get("value").tensor.dtype in constant.INPUT_FILE_TYPE:
    value_name = node_def.attr.get("value").tensor.string_val
    new_value_name = list()
    for value in value_name:
      replica_value_name = True
      for data_format in constant.PAI_DATA_FORMAT:
        if value.startswith(data_format):
          replica_value_name = False
          break
      if replica_value_name:
        value = replica_prefix + value
      new_value_name.append(value)
    node_def.attr.get('value').tensor.string_val[:] = new_value_name

  op_def = copy.deepcopy(orig_op.op_def)
  output_types = orig_op.output_types[:]
  input_types = orig_op.input_types[:]

  graph.unready_inputs_cache[node_def.name] = dict()
  graph.unready_control_inputs_cache[node_def.name] = dict()

  # get inputs
  inputs = []
  for inp_idx, inp in enumerate(orig_op.inputs):
    name = inp.name \
        if ((not graph.clone_dataset_related_ops) and graph.is_dataset_related(inp)) \
        else (replica_prefix + inp.name)
    if name in graph.tensors:
      inputs.append(graph.get_tensor_by_name(name).primitive_obj)
    else:
      tensor = graph.get_tensor_by_name(inp.name)
      if tensor.producer.get_control_flow_context() is not None:
        graph.original_context_cache[tensor.producer.name] = tensor.producer.get_control_flow_context()
        tensor.producer.set_control_flow_context(None)
      inputs.append(tensor.primitive_obj)
      graph.unready_inputs_cache[node_def.name][inp_idx] = name

  # get control inputs
  control_inputs = []
  old_control_inputs = list(orig_op.control_inputs)
  for c_inp in old_control_inputs:
    name = c_inp.name \
        if (graph.is_dataset_related(c_inp) and \
            (not graph.clone_dataset_related_ops)) \
        else (replica_prefix + c_inp.name)
    if name in graph.operations:
      control_inputs.append(graph.get_operation_by_name(name).primitive_obj)
    else:
      op = graph.get_operation_by_name(c_inp.name)
      if op.get_control_flow_context() is not None:
        graph.original_context_cache[op.name] = op.get_control_flow_context()
        op.set_control_flow_context(None)
      control_inputs.append(op.primitive_obj)
      graph.unready_control_inputs_cache[node_def.name][op.name] = name

  if not graph.unready_inputs_cache[node_def.name]:
    del graph.unready_inputs_cache[node_def.name]
  if not graph.unready_control_inputs_cache[node_def.name]:
    del graph.unready_control_inputs_cache[node_def.name]

  with ModelPhase(orig_op.phase):
    with ops.device(device):
      graph.current_cloned_taskgraph = orig_op.taskgraph
      new_op = ops.Operation(node_def,
                             ops.get_default_graph(),
                             inputs,
                             output_types,
                             control_inputs,
                             input_types,
                             None,
                             op_def=op_def)
      new_op._set_device(device)  # pylint: disable=protected-access


def context_def_clone(graph, context_def, is_while_context, replica_idx,
                      micro_batch_idx):
  """Clone control flow context from context_def
  with replica_idx and micto_batch_idx."""
  def fetch_prefix(tensor_name):
    """Fetch prefix for tensor in new_context_def."""
    op = graph.get_tensor_by_name(tensor_name).producer
    if graph.current_model_phase == ModelPhase.MICRO_BATCH_CLONE:
      prefix = micro_batch_prefix if graph.need_clone(op) else ""
    elif graph.current_model_phase == ModelPhase.REPLICATED:
      prefix = replica_prefix if graph.need_clone(op) else ""
      with ModelPhase(ModelPhase.MICRO_BATCH_CLONE):
        prefix += micro_batch_prefix if graph.need_clone(op) else ""
    else:
      raise RuntimeError(
          "ModelPhase {} is not supported for context def cloning.".format(
              graph.current_model_phase))
    return prefix

  if context_def.context_name.find('global_step') >= 0:
    return None

  if context_def.context_name not in graph.control_flow_context_map:
    graph.control_flow_context_map[context_def.context_name] = {}
  else:
    if micro_batch_idx in \
        graph.control_flow_context_map[context_def.context_name]:
      return None
  replica_prefix = common.get_replica_prefix(replica_idx)
  micro_batch_prefix = common.get_micro_batch_prefix(micro_batch_idx)
  prefix = replica_prefix + micro_batch_prefix
  new_context_def = control_flow_pb2.WhileContextDef() if is_while_context \
      else control_flow_pb2.CondContextDef()
  new_context_def.context_name = prefix + context_def.context_name
  new_context_def.pivot_name = \
      fetch_prefix(context_def.pivot_name) + context_def.pivot_name

  if is_while_context:
    new_context_def.parallel_iterations = context_def.parallel_iterations
    if context_def.maximum_iterations_name:
      new_context_def.maximum_iterations_name = fetch_prefix(context_def.maximum_iterations_name) + context_def.maximum_iterations_name
    new_context_def.back_prop = context_def.back_prop
    new_context_def.swap_memory = context_def.swap_memory
    new_context_def.pivot_for_pred_name = fetch_prefix(context_def.pivot_for_pred_name) + context_def.pivot_for_pred_name
    new_context_def.pivot_for_body_name = fetch_prefix(context_def.pivot_for_body_name) + context_def.pivot_for_body_name
    new_context_def.loop_exit_names.extend(
        [fetch_prefix(name) + name for name in context_def.loop_exit_names])
    new_context_def.loop_enter_names.extend([(fetch_prefix(name) + name) for name in context_def.loop_enter_names])
  else:
    new_context_def.pred_name = fetch_prefix(context_def.pred_name) + context_def.pred_name
    new_context_def.branch = context_def.branch

  values_def = control_flow_pb2.ValuesDef()
  values_def.values.extend([(fetch_prefix(name) + name) for name in context_def.values_def.values])
  for k, v in list(context_def.values_def.external_values.items()):
    values_def.external_values[fetch_prefix(k) + k] = fetch_prefix(v) + v
  new_context_def.values_def.MergeFrom(values_def)
  for nested in context_def.nested_contexts:
    new_nested = new_context_def.nested_contexts.add()
    if nested.while_ctxt.context_name:
      while_ctxt = context_def_clone(graph, nested.while_ctxt, True, replica_idx, micro_batch_idx)
      if while_ctxt is not None:
        new_nested.while_ctxt.CopyFrom(while_ctxt)
        new_context = control_flow_ops.WhileContext(context_def=while_ctxt)
        graph.control_flow_context_map[
            nested.while_ctxt.context_name][micro_batch_idx] = new_context
    elif nested.cond_ctxt.context_name:
      cond_ctxt = context_def_clone(graph, nested.cond_ctxt, False, replica_idx, micro_batch_idx)
      if cond_ctxt is not None:
        new_nested.cond_ctxt.CopyFrom(cond_ctxt)
        new_context = control_flow_ops.CondContext(context_def=cond_ctxt)
        graph.control_flow_context_map[
            nested.cond_ctxt.context_name][micro_batch_idx] = new_context
    else:
      raise TypeError(
          "Type of {} is not supported for nested control flow context.".
          format(nested))
  return new_context_def


def function_clone(graph, orig_func, cloned_func_name, target_device):
  """Clone function for a micro batch or a replica."""
  with ops.device(target_device):
    if Version(__version__) >= Version("1.12.0") and Version(__version__) < Version("1.14.0"):
      new_func = _DefinedFunction(func=orig_func.func,
                                  argnames=orig_func.arg_names,
                                  input_types=orig_func.input_types,
                                  func_name=cloned_func_name,
                                  grad_func=orig_func.grad_func,
                                  python_grad_func=orig_func.python_grad_func,
                                  out_names=orig_func.out_names,
                                  shape_func=orig_func.shape_func,
                                  capture_by_value=orig_func.capture_by_value)
    elif Version(__version__) < Version("2.0"):
      new_func = _DefinedFunction(
          func=orig_func.func,
          argnames=orig_func.arg_names,
          input_types=orig_func.input_types,
          func_name=cloned_func_name,
          grad_func=orig_func.grad_func,
          python_grad_func=orig_func.python_grad_func,
          out_names=orig_func.out_names,
          shape_func=orig_func.shape_func,
          capture_by_value=orig_func.capture_by_value,
          whitelisted_stateful_ops=orig_func.whitelisted_stateful_ops,
          capture_resource_var_by_value=orig_func.capture_resource_var_by_value)
    new_func.add_to_graph(ops.get_default_graph())
  new_func = graph.get_function_by_name(cloned_func_name)
  new_func.is_dataset_related = orig_func.is_dataset_related
  new_func.taskgraph = orig_func.taskgraph
  orig_func.taskgraph.add_functions(new_func)


def control_flow_context_clone(graph, replica_idx, num_micro_batch):
  """Clone control flow context for a model replica."""
  for micro_batch_idx in range(num_micro_batch):
    for context in ops.get_collection(ops.GraphKeys.WHILE_CONTEXT):
      context_def = context.to_proto()
      new_context_def = context_def_clone(graph, context_def, True, replica_idx, micro_batch_idx)
      if new_context_def is None:
        continue
      new_context = control_flow_ops.WhileContext(context_def=new_context_def)
      graph.control_flow_context_map[context_def.context_name][micro_batch_idx] = new_context
    for context in ops.get_collection(ops.GraphKeys.COND_CONTEXT):
      context_def = context.to_proto()
      new_context_def = context_def_clone(graph, context_def, False, replica_idx, micro_batch_idx)
      if new_context_def is None:
        continue
      new_context = control_flow_ops.CondContext(context_def=new_context_def)
      graph.control_flow_context_map[context_def.context_name][micro_batch_idx] = new_context


def grad_loop_state_clone_internel(graph, op_list, replica_idx, micro_batch_idx):
  """Clone grad loop state for op_list."""
  for op in op_list:
    op_context = op.get_control_flow_context()
    context_def_list = []
    if op_context is None:
      continue
    grad_state = op_context.grad_state
    if grad_state:
      context_def_list.append(grad_state.forward_context.to_proto())
      context_def_list.append(grad_state.grad_context.to_proto())
      out_grad_state = grad_state.outer_grad_state
      if out_grad_state:
        context_def_list.append(out_grad_state.forward_context.to_proto())
        context_def_list.append(out_grad_state.grad_context.to_proto())

    for context_def in context_def_list:
      new_context_def = context_def_clone(graph, context_def, True, replica_idx, micro_batch_idx)
      if new_context_def is not None:
        new_context = control_flow_ops.WhileContext(context_def=new_context_def)
        graph.control_flow_context_map[context_def.context_name][micro_batch_idx] = new_context


def grad_loop_state_clone(graph, replica_idx, taskgraph, num_micro_batch):
  """Clone grad loop state for a replica or multi micro-batches."""
  start_micro_batch_idx = 1 if graph.current_model_phase == ModelPhase.MICRO_BATCH_CLONE else 0
  for micro_batch_idx in range(start_micro_batch_idx, num_micro_batch):
    op_list = taskgraph.operations.forward_operations(0, 0) + taskgraph.operations.backward_operations(0, 0)
    grad_loop_state_clone_internel(graph, op_list, replica_idx, micro_batch_idx)
  op_list = taskgraph.operations.apply_operations(0) + taskgraph.operations.save_and_restore_operations(0)
  grad_loop_state_clone_internel(graph, op_list, replica_idx, 0)


def add_control_dependency(ops_first, ops_second):
  """Sequence `ops_first` before `ops_second`.

  Given two operations, returns a pair of operations with the same behavior
  except that the first returned operation will execute before the second
  returned operation.
  """
  if not isinstance(ops_first, list):
    ops_first = [ops_first]
  if not isinstance(ops_second, list):
    ops_second = [ops_second]
  for op in ops_second:
    op.add_control_inputs(ops_first)


def create_communicator(name, devices):
  """Create a communicator with parameters from env."""
  conf = Env.get().config
  return CollectiveCommunicator(
      name=name,
      devices=devices,
      max_splits=conf.communication.max_splits,
      num_communicators=conf.communication.num_communicators,
      enable_fp16=conf.communication.fp16,
      fp16_scale=conf.communication.fp16_scale)


def create_serial_communicator(name, devices, max_splits=None):
  """Create a serial communicator with num_communicators=1.
     Using for broadcast variables and so on."""
  if max_splits is None:
    num_splits = constant.DEFAULT_COM_MAX_SPLITS
  else:
    num_splits = max_splits
  return CollectiveCommunicator(name=name,
                                devices=devices,
                                max_splits=num_splits,
                                num_communicators=1)


def create_simple_communicator(name, devices):
  """Create a simple communicator with max_splits=1.
     Using for summarizing loss, acc and so on."""
  return CollectiveCommunicator(name=name,
                                devices=devices,
                                max_splits=1)


def allreduce_gradients(gradients,
                        current_device,
                        all_devices,
                        comm_index,
                        mean=False,
                        name="GRADIENT_REDUCE"):
  """AllReduce gradients for all replicas."""
  comm_name = name + "_{}"
  with ops.device(current_device):
    comm = create_communicator(name=comm_name.format(comm_index),
                               devices=all_devices)
    return comm.batch_allreduce(gradients, mean=mean)


def allreduce_tensors(comm, tensors, current_device, mean=False):
  """AllReduce tensors for all replicas."""
  with ops.device(current_device):
    return comm.batch_allreduce(tensors, mean=mean)


def allgather_tensors(comm, tensors, current_device, mean=None):
  """Allgather tensors for all replicas."""
  if mean:
    raise ValueError("Mean in allgather expected None/False, while %s." % mean)
  comm_tensors = []
  with ops.device(current_device):
    for tensor in tensors:
      comm_tensors.append(comm.allgather(tensor))
  return comm_tensors


def alltoall(comm, tensor, current_device):
  """AlltoAll tensors."""
  conf = Env.get().config
  if conf.communication.fp16 and tensor.dtype == dtypes.float32:
    tensor_float16 = math_ops.cast(tensor, dtypes.float16)
    with ops.device(current_device):
      result = comm.alltoall(tensor_float16)
      return math_ops.cast(result, dtypes.float32)
  else:
    with ops.device(current_device):
      return comm.alltoall(tensor)


def concat_indexed_slices(tensor_list, dense_shape):
  """Concat IndexedSlices."""
  values_list, indices_list = list(
      zip(*[(t.values, t.indices) for t in tensor_list]))
  return ops.IndexedSlices(array_ops.concat(values_list, axis=0),
                           array_ops.concat(indices_list, axis=0),
                           dense_shape=dense_shape)


def dispatch_across_consumers(products, consumers, rank, even=False):
  """Dispatch products across all consumers.

    Args:
      products: number of products to be dispatched.
      consumers: number of consumers.
      rank: rank of this consumer in all consumers.
      even: dispatch across consumers evenly if True.
  """
  if even:
    assert products % consumers == 0, \
        "Number of products expects to be divided by number of consumers" \
        ". while products = {}, consumers = {}.".format(products, consumers)
  remainder = products % consumers
  divisor = products // consumers
  return (divisor + 1) if rank < remainder \
      else divisor

def node_clone_for_amp(orig_op, new_type, new_type_pb, suffix):
  """Clone a operation to 'device' from 'orig_op' for amp."""
  # get node def
  device = orig_op.device
  node_def = copy.deepcopy(orig_op.node_def)
  loc_attr = node_def.attr.get('_class')
  node_def.name = node_def.name + suffix + '_' + new_type.name
  node_def.attr.get('T').type = new_type_pb
  if loc_attr:
    for idx in range(len(loc_attr.list.s)):
      loc_attr.list.s[idx] = \
          loc_attr.list.s[idx].replace(
              b'@', b'@' + compat.as_bytes(suffix))
  for attr_name in ["frame_name", "shared_name"]:
    attr_value = node_def.attr.get(attr_name)
    if attr_value and attr_value.s:
      node_def.attr.get(attr_name).s = attr_value.s + compat.as_bytes(suffix)

  op_def = copy.deepcopy(orig_op.op_def)

  output_types = [new_type_pb for _ in orig_op.output_types]
  inputs = [inp.primitive_obj for inp in orig_op.inputs]
  new_inputs = []
  input_types = []
  for inp in inputs:
    if inp.dtype != new_type and inp.dtype in [dtypes.float32, dtypes.float16]:
      new_input = math_ops.cast(inp, new_type, name=inp.op.name + suffix + '_cast_' + new_type.name)
      new_inputs.append(new_input)
      input_types.append(new_type)
    else:
      new_inputs.append(inp)
      input_types.append(inp.dtype)
  control_inputs = [t for t in list(orig_op.control_inputs)]

  with ModelPhase(orig_op.phase):
    with ops.device(device):
      new_op = ops.Operation(node_def,
                             ops.get_default_graph(),
                             new_inputs,
                             output_types,
                             control_inputs,
                             input_types,
                             None,
                             op_def=op_def)
      new_op._set_device(device)  # pylint: disable=protected-access
      return new_op
