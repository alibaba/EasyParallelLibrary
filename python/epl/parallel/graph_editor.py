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
""""Classes to transform original graph into parallel graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random

from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops as tfops
from tensorflow.python.training import queue_runner
from tensorflow.python.platform import tf_logging
from tensorflow.python.util import compat

from epl.env import Env
from epl.ir.graph import Graph
from epl.ir.phase import ModelPhase
from epl.parallel.ops import allreduce_gradients
from epl.parallel.ops import concat_indexed_slices
from epl.parallel.ops import control_flow_context_clone
from epl.parallel.ops import function_clone
from epl.parallel.ops import grad_loop_state_clone
from epl.parallel.ops import node_clone_for_pipeline
from epl.parallel.ops import node_clone_for_replicas
from epl.runtime.zero import zero_v0
from epl.strategies.scheduler import get_scheduler
from epl.strategies.replicate import Replicate
from epl.utils import common
from epl.utils import constant


class Custom(object):
  """Class to fetch forward_entrance_ops, forwrad_exit_ops,
  backward_entrance_ops, backward_exit_ops of a taskgraph for
  one replica."""
  def __init__(self, taskgraph, replica_idx):
    self._taskgraph = taskgraph
    self._replica_idx = replica_idx
    self._num_micro_batch = taskgraph.pipeline_config.num_micro_batch
    self._forward_entrance_ops = []
    self._forward_exit_ops = []
    self._backward_entrance_ops = []
    self._backward_exit_ops = []
    self.__call__()

  def __call__(self):
    """Function to fetch forward_entrance_ops, forwrad_exit_ops,
    backward_entrance_ops, backward_exit_ops from parallel graph."""
    for micro_batch_idx in range(self._num_micro_batch):
      self._forward_entrance_ops.append(
          list(self._taskgraph.forward_entrance_ops(self._replica_idx,
                                                    micro_batch_idx)))
      self._forward_exit_ops.append(
          list(self._taskgraph.forward_exit_ops(self._replica_idx,
                                                micro_batch_idx)))
      self._backward_entrance_ops.append(
          list(self._taskgraph.backward_entrance_ops(self._replica_idx,
                                                     micro_batch_idx)))
      self._backward_exit_ops.append(
          list(self._taskgraph.backward_exit_ops(self._replica_idx,
                                                 micro_batch_idx)))

  @property
  def forward_entrance_ops(self):
    return self._forward_entrance_ops

  @property
  def forward_exit_ops(self):
    return self._forward_exit_ops

  @property
  def backward_entrance_ops(self):
    return self._backward_entrance_ops

  @property
  def backward_exit_ops(self):
    return self._backward_exit_ops

  def serialize(self):
    return "epl.Custom(index = %s, forward_entrance_ops=%s, \
        forward_exit_ops=%s, backward_entrance_ops=%s, \
        backward_exit_ops=%s)" \
        % (self._taskgraph.index, self._forward_entrance_ops, \
        self._forward_exit_ops, self._backward_entrance_ops, \
        self._backward_exit_ops)

  def __str__(self):
    return self.serialize()

  def __repr__(self):
    return self.serialize()


class GraphEditor(object):
  """Classes of operators to transform original graph into parallel graph."""

  @property
  def _graph(self):
    return Graph.get()

  def table_io_slicing(self, dataset_api_op):
    """Slicing table to balance data load among all model replicas."""
    slice_id = 0
    all_devices = dataset_api_op.taskgraph.virtual_device.all_devices
    list.sort(all_devices)
    if self._graph.num_constructors > 1:
      total_num_slices = len(all_devices)
      for idx, dev in enumerate(all_devices):
        if dev == dataset_api_op.device:
          slice_id = idx
          break
    else:
      total_num_slices = 1
    if dataset_api_op.node_def.attr.get("slice_id") and \
        dataset_api_op.node_def.attr.get("slice_count"):
      node_def = dataset_api_op.node_def
      node_def.attr.get("slice_count").i = total_num_slices
      node_def.attr.get("slice_id").i = slice_id
      dataset_api_op.set_attr("slice_count", node_def.attr.get("slice_count"))
      dataset_api_op.set_attr("slice_id", node_def.attr.get("slice_id"))
    else:
      inputs = dataset_api_op.primitive_obj.inputs
      for inp in inputs:
        inp = self._graph.get_function_tensor_by_name(inp.name)
        if inp.name == "slice_id:0":
          node_def = inp.producer.node_def
          node_def.attr.get("value").i = slice_id
          inp.producer.set_attr("value", node_def.attr.get("value"))
        elif inp.name.endswith("slice_count:0"):
          node_def = inp.producer.node_def
          node_def.attr.get("value").i = total_num_slices
          inp.producer.set_attr("value", node_def.attr.get("value"))

  def io_slicing(self):
    """Slicing files to each contructor in proportion
    to num_local_replicas."""
    # TODO(jiangle.jl): Support more dataset api, including placeholder.
    if not self._graph.dataset_api_op:
      return

    for dataset_api_op in self._graph.dataset_api_op:
      if dataset_api_op.type in constant.ODPS_TABLE_API_OPS:
        self.table_io_slicing(dataset_api_op)
        continue
      if self._graph.num_constructors <= 1:
        continue
      if dataset_api_op.inputs:
        file_op = dataset_api_op.inputs[0].producer
      else:
        name = dataset_api_op.name + "/Const"
        if name in self._graph.operations:
          file_op = \
              self._graph.get_operation_by_name(name)
        else:
          continue
      node_def = file_op.node_def
      if node_def.attr.get("value") is None or \
          node_def.attr.get("value").tensor is None or \
          node_def.attr.get("value").tensor.dtype not in constant.INPUT_FILE_TYPE:
        continue
      file_name_list = node_def.attr.get("value").tensor.string_val
      if not file_name_list:
        continue

      consumers_inputs_op = []
      for obj in list(dataset_api_op.outputs):
        consumers = self._graph.get_tensor_by_name(obj.name).consumers
        for consumer in consumers:
          inputs = [inp.producer for inp in consumer.inputs]
          consumers_inputs_op += inputs
      continue_io_slicing = True
      for inp in consumers_inputs_op:
        if inp.type in constant.ODPS_TABLE_API_OPS:
          continue_io_slicing = False
          break
      if not continue_io_slicing:
        continue

      taskgraph = dataset_api_op.taskgraph
      all_devices = taskgraph.virtual_device.all_devices
      if taskgraph.num_replicas <= 1:
        continue
      slice_files = \
          fetch_slice_objects_proportion_to_local_num_replicas(
              Env.get().cluster.worker_index,
              file_name_list,
              taskgraph.num_replicas,
              all_devices,
              Env.get().config.io.drop_last_files,
              Env.get().config.io.unbalanced_io_slicing,
              name="dataset files")
      node_def.attr.get("value").tensor.string_val[:] = slice_files
      dim_size = len(node_def.attr.get("value").tensor.string_val)
      node_def.attr.get('value').tensor.tensor_shape.Clear()
      node_def.attr.get('value').tensor.tensor_shape.dim.append(
          node_def.attr.get('value').tensor.tensor_shape.Dim(size=dim_size))
      file_op.set_attr("value", node_def.attr.get("value"))
      tf_logging.info(
          "Current constructor handles {} files including {}" \
          .format(len(slice_files), slice_files))

  def get_target_device_for_clone(self, orig_op_or_func, replica_idx):
    """Get target device for new_op cloned form orig_op or orig_function."""
    local_devices = orig_op_or_func.taskgraph.virtual_device.local_devices
    if not local_devices:
      local_devices = self._graph.taskgraphs[-2].virtual_device.local_devices
    target_device = local_devices[replica_idx]
    if common.get_device_type(orig_op_or_func.device) == constant.HOST_DEVICE:
      return common.get_device_string(task=Env.get().cluster.worker_index, device_type="CPU")
    if orig_op_or_func.device != local_devices[0]:
      for stage_idx in range(self._graph.num_stages):
        if orig_op_or_func.device == \
            self._graph.taskgraphs[stage_idx].virtual_device.local_devices[0]:
          target_device = \
              self._graph.taskgraphs[stage_idx].virtual_device.local_devices[replica_idx]
          break
    return target_device

  def op_device_replacement(self, op):
    """Device replacement of one operation."""
    if op.device_already_set:
      return
    loc_str = b"loc:@"
    loc_attr = op.node_def.attr.get("_class")
    loc_device = None
    if loc_attr:
      for idx in range(len(loc_attr.list.s)):
        if loc_attr.list.s[idx].startswith(loc_str):
          loc_op_name = loc_attr.list.s[idx][len(loc_str):]
          loc_op_name = compat.as_str(loc_op_name, constant.ENCODING)
          if loc_op_name == op.name:
            break
          loc_op = self._graph.get_operation_by_name(loc_op_name)
          self.op_device_replacement(loc_op)
          loc_device = self._graph.get_operation_by_name(loc_op_name).device
    if loc_device:
      op.set_device(loc_device)
    else:
      op.set_device(op.taskgraph.virtual_device.local_devices[0])

  # pylint: disable=protected-access
  def partition_stages(self, stage_ops):
    """
    partition graph to taskgraphs based on cost model
    """
    if len(stage_ops) >= 1:
      assert len(self._graph.taskgraphs) == 1, \
          "Current only support tasks with only one taskgraph"
      taskgraph = self._graph.taskgraphs[0]
      # Create new taskgraphs for partitioned results
      # Append newly created taskgraphs to current graph
      # In the end, remove the original taskgraph

      # Since we append new taskgraphs after the original taskgraph,
      # to match the taskgraph index, we need to insert the one slice devices
      virtual_devices = Env.get().cluster.virtual_devices
      virtual_devices.insert(0, virtual_devices[0])
      for ops in stage_ops:
        strategy = Replicate(device_count=1)
        self._graph.strategy_context.add_context(strategy)
        self._graph.strategy_context.update_flag = True
        for op in ops:
          self._graph.add_operation(op.primitive_obj)
        self._graph.strategy_context.del_context(strategy)
    # Remove old taskgraph.
    del self._graph.taskgraphs[0]
    assert self._graph.num_stages == len(stage_ops), \
        "Taskgraph number {} is not equal to number of auto stage_ops {}" \
        .format(self._graph.num_stages, len(stage_ops))
    for taskgraph in self._graph._taskgraphs:
      taskgraph._index -= 1

  def device_replacement(self):
    """Reset device info of forward operations in original model replica."""
    for taskgraph in self._graph.taskgraphs:
      for op in taskgraph.operations.forward_operations(0, 0):
        self.op_device_replacement(op)
    for op in self._graph.operations.values():
      for colocate_name in op.primitive_obj.colocation_groups():
        if self._graph.colocation_groups[colocate_name]:
          colocate_device = self._graph.colocation_groups[colocate_name][0].device
          if op.device != colocate_device:
            op.set_device(colocate_device)
            break

  def _update_inputs_or_control_inputs(self):
    """Update inputs or control_inputs for op in while_loop
    in a micro-batch model."""
    for name, inputs_map in list(self._graph.unready_inputs_cache.items()):
      for inp_idx, inp_name in list(inputs_map.items()):
        new_op = self._graph.get_operation_by_name(name)
        new_op.update_input(
            inp_idx,
            self._graph.get_tensor_by_name(inp_name).primitive_obj)
    for name, control_inputs_map in \
        list(self._graph.unready_control_inputs_cache.items()):
      new_op = self._graph.get_operation_by_name(name)
      control_inputs = []
      old_control_inputs = list(new_op.control_inputs)
      for c_inp in old_control_inputs:
        if c_inp.name in control_inputs_map:
          c_inp_op = self._graph.get_operation_by_name(control_inputs_map[c_inp.name])
          if c_inp_op is None:
            raise RuntimeError("Cannot get control_input {} of operation {} by name".format(control_inputs_map[c_inp.name], new_op))
          else:
            control_inputs.append(c_inp_op.primitive_obj)
        else:
          control_inputs.append(c_inp)
      new_op.control_inputs = control_inputs
    self._graph.reset_cache_for_clone()

  def _forward_clone(self):
    """Clone forward part of all taskgraphs for other replicas."""
    for taskgraph in self._graph.taskgraphs:
      if taskgraph.local_num_replicas <= 1:
        continue
      for replica_idx in range(1, taskgraph.local_num_replicas):
        for micro_batch_idx in \
            range(taskgraph.pipeline_config.num_micro_batch):
          for ele in \
              taskgraph.operations.forward_operations(0, micro_batch_idx):
            if self._graph.need_clone(ele):
              target_device = self.get_target_device_for_clone(
                  ele, replica_idx)
              node_clone_for_replicas(self._graph, ele, replica_idx,
                                      target_device)

  def _backward_clone(self):
    """Clone backward part of all taskgraphs for other replicas."""
    for taskgraph_idx, _ in enumerate(self._graph.taskgraphs):
      reversed_idx = self._graph.num_stages - 1 - taskgraph_idx
      current_taskgraph = self._graph.taskgraphs[reversed_idx]
      if current_taskgraph.local_num_replicas <= 1:
        continue
      for replica_idx in range(1, current_taskgraph.local_num_replicas):
        for micro_batch_idx in \
            range(current_taskgraph.pipeline_config.num_micro_batch):
          backward_operations_to_clone = \
              current_taskgraph.operations. \
                  backward_operations(0, micro_batch_idx)
          for ele in backward_operations_to_clone:
            if self._graph.need_clone(ele):
              target_device = \
                  self.get_target_device_for_clone(ele, replica_idx)
              node_clone_for_replicas(self._graph, ele, replica_idx,
                                      target_device)

  def _apply_clone(self):
    """Clone apply part of all taskgraphs for other replicas."""
    for taskgraph in self._graph.taskgraphs:
      if taskgraph.local_num_replicas <= 1:
        continue
      for replica_idx in range(1, taskgraph.local_num_replicas):
        for ele in taskgraph.operations.apply_operations(0):
          if self._graph.need_clone(ele):
            target_device = \
                self.get_target_device_for_clone(ele, replica_idx)
            node_clone_for_replicas(self._graph, ele, replica_idx,
                                    target_device)

  def _save_and_restore_clone(self):
    """Clone save_and_restore part of all taskgraphs for other replicas."""
    # Clone save_and_store graph only once for each replica.
    for taskgraph in self._graph.taskgraphs:
      for replica_idx in range(1, taskgraph.local_num_replicas):
        for ele in taskgraph.operations.save_and_restore_operations(0):
          if self._graph.need_clone(ele):
            target_device = self.get_target_device_for_clone(ele, replica_idx)
            node_clone_for_replicas(self._graph, ele, replica_idx,
                                    target_device)

  def _clone_micro_batch_internel(self, micro_batch_idx, operations):
    """Wrapper function to clone one taskgraph."""
    with ModelPhase(ModelPhase.MICRO_BATCH_CLONE):
      for ele in operations:
        if self._graph.need_clone(ele):
          node_clone_for_pipeline(self._graph, ele, micro_batch_idx,
                                  ele.device)

  def micro_batch_clone(self):
    """Micro-batch clone of Taskgraph for the first replica."""
    if not self._graph.pipeline_enabled:
      return
    with ModelPhase(ModelPhase.MICRO_BATCH_CLONE):
      # Clone forward graph for a micro batch model.
      for taskgraph_idx, taskgraph in enumerate(self._graph.taskgraphs):
        forward_operations = taskgraph.operations.forward_operations(0, 0)
        for micro_batch_idx in \
            range(1, taskgraph.pipeline_config.num_micro_batch):
          self._clone_micro_batch_internel(micro_batch_idx, forward_operations)
      # Clone backward graph for a micro batch model.
      for taskgraph_idx, taskgraph in enumerate(self._graph.taskgraphs):
        reversed_idx = self._graph.num_stages - 1 - taskgraph_idx
        current_taskgraph = self._graph.taskgraphs[reversed_idx]
        backward_operations = \
            current_taskgraph.operations.backward_operations(0, 0)
        for micro_batch_idx in \
            range(1, current_taskgraph.pipeline_config.num_micro_batch):
          self._clone_micro_batch_internel(micro_batch_idx,
                                           backward_operations)
      self._update_inputs_or_control_inputs()
      self._fix_control_flow_context_for_micro_batches()
      # Clone all function definition for micro batches
      self._clone_function_for_micro_batches()

  def replicas_clone(self):
    """Clone other replicas."""
    with ModelPhase(ModelPhase.REPLICATED):
      # Replicated forward operations.
      self._forward_clone()
      # Replicated backward operations.
      self._backward_clone()
      # Replicated apply operations.
      self._apply_clone()
      # Replicated save_and_restore operations.
      self._save_and_restore_clone()
      # Update inputs or control_inputs for cloned_op.
      self._update_inputs_or_control_inputs()
      # Fix control flow context for replicas
      self._fix_control_flow_context_for_replicas()
      # Clone all function definition for replicas
      self._clone_function_for_replicas()
      # Update function for replicas
      self._fix_function_definition()
      # Fix queue runners.
      self._fix_queue_runners()

  def _fix_queue_runners(self):
    """Clone queue runners for other replicas."""
    if not self._graph.clone_dataset_related_ops:
      return
    for taskgraph in self._graph.taskgraphs:
      if taskgraph.local_num_replicas <= 1:
        continue
      queue_runners = taskgraph.queue_runners
      for replica_idx in range(1, taskgraph.local_num_replicas):
        replica_prefix = common.get_replica_prefix(replica_idx)
        for queue_run in queue_runners:
          queue = queue_run.queue
          enq_ops = \
              [self._graph.get_operation_by_name(
                  replica_prefix + op.name).primitive_obj \
               for op in queue_run.enqueue_ops]
          cal_op = \
              self._graph.get_operation_by_name(
                  replica_prefix + queue_run.cancel_op.name).primitive_obj
          queue_runner.add_queue_runner(
              queue_runner.QueueRunner(queue,
                                       enqueue_ops=enq_ops,
                                       cancel_op=cal_op))

  def _fix_function_definition(self):
    """Update function definition of cloned_op."""
    for op, attr in list(self._graph.op_with_function_map.items()):
      func_name = op.node_def.attr.get(attr).func.name
      op_or_func_name = op.name if op.function is None else op.function.name
      replica_idx = \
          common.get_replica_index_from_node_name(op_or_func_name)
      micro_batch_idx = \
          common.get_micro_batch_index_from_node_name(op_or_func_name)
      if not replica_idx and not micro_batch_idx:
        continue
      replica_prefix = common.get_replica_prefix(replica_idx)
      micro_batch_prefix = common.get_micro_batch_prefix(micro_batch_idx)
      cloned_func_name = \
          compat.as_str(replica_prefix + micro_batch_prefix + func_name, constant.ENCODING)
      node_def = copy.deepcopy(op.node_def)
      node_def.attr.get(attr).func.name = cloned_func_name
      op.set_attr(attr, node_def.attr.get(attr))

  def _clone_function_for_micro_batches(self):
    """Clone DefinedFunction for micro batches."""
    for taskgraph in self._graph.taskgraphs:
      current_funcs = list(taskgraph.functions)
      for function in current_funcs:
        if common.get_micro_batch_index_from_node_name(function.name):
          continue
        if function.is_dataset_related:
          continue
        for micro_batch_idx in \
            range(1, taskgraph.pipeline_config.num_micro_batch):
          micro_batch_prefix = common.get_micro_batch_prefix(micro_batch_idx)
          cloned_func_name = compat.as_str(micro_batch_prefix + function.name, constant.ENCODING)
          if cloned_func_name in self._graph.functions:
            continue
          function_clone(self._graph, function, cloned_func_name,
                         function.device)

  def _clone_function_for_replicas(self):
    """Clone DefinedFunction for replicas."""
    for taskgraph in self._graph.taskgraphs:
      current_funcs = list(taskgraph.functions)
      for function in current_funcs:
        if common.get_replica_index_from_node_name(function.name):
          continue
        if (not self._graph.clone_dataset_related_ops) and \
            function.is_dataset_related:
          continue
        for replica_idx in range(1, taskgraph.local_num_replicas):
          replica_prefix = common.get_replica_prefix(replica_idx)
          cloned_func_name = compat.as_str(replica_prefix + function.name, constant.ENCODING)
          if cloned_func_name in self._graph.functions:
            continue
          target_device = \
              self.get_target_device_for_clone(function, replica_idx)
          function_clone(self._graph, function, cloned_func_name,
                         target_device)

  def _fix_control_flow_context_internel(self,
                                         op_list,
                                         micro_batch_idx=0,
                                         replica_idx=0):
    """Fix control flow context for all operations in op_list."""
    micro_batch_prefix = common.get_micro_batch_prefix(micro_batch_idx)
    replica_prefix = common.get_replica_prefix(replica_idx)
    prefix = replica_prefix \
        if self._graph.current_model_phase == ModelPhase.REPLICATED \
        else micro_batch_prefix
    for op in op_list:
      old_context = op.get_control_flow_context()
      if old_context is None and op.name in self._graph.original_context_cache:
        old_context = self._graph.original_context_cache[op.name]
        op.set_control_flow_context(old_context)
      if self._graph.need_clone(op) and old_context is not None:
        new_op = self._graph.get_operation_by_name(prefix + op.name)
        new_context_key = \
            common.get_original_name_from_cloned_object(
                old_context.to_proto().context_name)
        if new_context_key not in \
            self._graph.control_flow_context_map:
          continue
        if micro_batch_idx in \
            self._graph.control_flow_context_map[new_context_key]:
          new_context = \
              self._graph.control_flow_context_map[
                  new_context_key][micro_batch_idx]
          new_op.set_control_flow_context(new_context)

  def _fix_control_flow_context_for_micro_batches(self):
    """Fix control flow context for all micro-batch models in a replica."""
    with ModelPhase(ModelPhase.MICRO_BATCH_CLONE):
      for taskgraph in self._graph.taskgraphs:
        num_micro_batch = taskgraph.pipeline_config.num_micro_batch
        control_flow_context_clone(self._graph, 0, num_micro_batch)
        grad_loop_state_clone(self._graph, 0, taskgraph, num_micro_batch)
        for micro_batch_idx in range(1, num_micro_batch):
          op_list = taskgraph.operations.forward_operations(0, 0) + \
              taskgraph.operations.backward_operations(0, 0)
          self._fix_control_flow_context_internel(
              op_list, micro_batch_idx=micro_batch_idx)
        self._graph.control_flow_context_map = {}

  def _fix_control_flow_context_for_replicas(self):
    """Fix control flow context for other replicas."""
    with ModelPhase(ModelPhase.REPLICATED):
      for taskgraph in self._graph.taskgraphs:
        for replica_idx in range(1, taskgraph.local_num_replicas):
          num_micro_batch = taskgraph.pipeline_config.num_micro_batch
          if not self._graph.pipeline_enabled:
            num_micro_batch = 1
          control_flow_context_clone(self._graph, replica_idx, num_micro_batch)
          grad_loop_state_clone(self._graph, replica_idx, taskgraph,
                                num_micro_batch)
          for micro_batch_idx in range(num_micro_batch):
            op_list = \
                taskgraph.operations.forward_operations(0, micro_batch_idx) + \
                taskgraph.operations.backward_operations(0, micro_batch_idx)
            self._fix_control_flow_context_internel(
                op_list, micro_batch_idx=micro_batch_idx, \
                replica_idx=replica_idx)
          op_list = taskgraph.operations.apply_operations(0) + taskgraph.operations.save_and_restore_operations(0)
          self._fix_control_flow_context_internel(op_list,
                                                  replica_idx=replica_idx)
          self._graph.control_flow_context_map = {}

  def schedule_optimization(self):
    """Pipeline schedule optimization for all replicas."""
    if not self._graph.pipeline_enabled:
      return
    pipeline_config = self._graph.get_pipeline_config()
    scheduler = get_scheduler(pipeline_config.strategy)(pipeline_config.num_micro_batch, self._graph.num_stages)

    # TODO(wangang.wa): It's not a rebust way to get number of replicas
    # from first taskgraph.
    local_num_replicas = self._graph.taskgraphs[0].local_num_replicas
    for replica_idx in range(local_num_replicas):
      # Clarify a custom obj for every taskgraph.
      customs = []
      for taskgraph in self._graph.taskgraphs:
        customs.append(Custom(taskgraph, replica_idx))
      scheduler.call(customs)

  def _accumulate_one_replica(self, taskgraph, replica_idx, mean=False):
    """Accumulate gradients from micro-batch inputs of
    one taskgraph in a replica."""
    original_grads = []
    acc_grads = []
    replica_prefix = common.get_replica_prefix(replica_idx)
    for grad in taskgraph.gradients:
      if common.is_indexed_slices(grad):
        grad_vaules = self._graph.get_tensor_by_name(replica_prefix +
                                                     grad.values.name)
        grad_indices = self._graph.get_tensor_by_name(replica_prefix +
                                                      grad.indices.name)
        acc_grad = tfops.IndexedSlices(
            grad_vaules.primitive_obj,
            grad_indices.primitive_obj,
            dense_shape=grad.dense_shape)
        original_grads.append(grad_vaules)
        original_grads.append(grad_indices)
      else:
        original_grad = self._graph.get_tensor_by_name(replica_prefix +
                                                       grad.name)
        acc_grad = original_grad.primitive_obj
        original_grads.append(original_grad)

      if not common.is_indexed_slices(acc_grad):
        micro_batch_grads_dense = [acc_grad]

      if self._graph.pipeline_enabled:
        for micro_batch_idx in \
            range(1, taskgraph.pipeline_config.num_micro_batch):
          micro_batch_prefix = \
              common.get_micro_batch_prefix(micro_batch_idx)
          if common.is_indexed_slices(grad):
            other_grad_values = self._graph.get_tensor_by_name(
                replica_prefix + micro_batch_prefix + grad.values.name)
            other_grad_indices = self._graph.get_tensor_by_name(
                replica_prefix + micro_batch_prefix + grad.indices.name)
            other_grad = tfops.IndexedSlices(
                other_grad_values.primitive_obj,
                other_grad_indices.primitive_obj,
                dense_shape=grad.dense_shape)
          else:
            other_grad = self._graph.get_tensor_by_name(
                replica_prefix + micro_batch_prefix + grad.name).primitive_obj

          if common.is_indexed_slices(acc_grad):
            with tfops.device(acc_grad.device):
              acc_grad = concat_indexed_slices([acc_grad, other_grad],
                                               grad.dense_shape)
          else:
            micro_batch_grads_dense.append(other_grad)

        if not common.is_indexed_slices(acc_grad):
          with tfops.device(acc_grad.device):
            acc_grad = sum(micro_batch_grads_dense)
            if mean:
              acc_grad = acc_grad / taskgraph.pipeline_config.num_micro_batch
      acc_grads.append(acc_grad)
    return original_grads, acc_grads

  def gradient_aggregation(self):
    """Aggregate gradients from micro-batch inputs of all replicas."""
    def _update_consumer_inernel(tensor_first, tensor_second):
      """Update inputs of consumer attributes from tensor_first
      to tensor_second."""
      for consumer in tensor_first.consumers:
        # Is within parallel strategy.
        if constant.PARALLEL_STRATEGY in consumer.name:
          continue
        for in_idx, inp in enumerate(consumer.primitive_obj.inputs):
          if inp.name == tensor_first.name:
            consumer.primitive_obj._update_input(in_idx, tensor_second)

    def update_consumer_inputs(tensors_first, tensors_second):
      """Update inputs of consumer attributed for tensors."""
      index = 0
      for tensor in tensors_second:
        if common.is_indexed_slices(tensor):
          _update_consumer_inernel(tensors_first[index], tensor.values)
          index = index + 1
          _update_consumer_inernel(tensors_first[index], tensor.indices)
        else:
          _update_consumer_inernel(tensors_first[index], tensor)
        index = index + 1

    mean_grad_flag = True if Env.get().config.communication.gradients_reduce_method == \
                             constant.REDUCE_METHOD_MEAN else False
    for taskgraph_idx, taskgraph in enumerate(self._graph.taskgraphs):
      all_devices = taskgraph.virtual_device.all_devices
      num_replicas = taskgraph.num_replicas
      for replica_idx in range(taskgraph.local_num_replicas):
        original_grads, merged_grads = \
            self._accumulate_one_replica(taskgraph, replica_idx, mean_grad_flag)
        if not original_grads or not merged_grads:
          continue
        # Only allreduce gradients produced by operations under replicate strategy.
        if taskgraph.strategy_context.replicate_strategy:
          current_device = original_grads[0].device
          # Call all_reduce for gradient aggregation of a taskgraph.
          merged_grads = merged_grads if num_replicas == 1 \
              else allreduce_gradients(merged_grads,
                                       current_device,
                                       all_devices,
                                       taskgraph_idx,
                                       mean_grad_flag,
                                       name="DATA_PARALLEL_GRADS_REDUCE")

        # Update input of gradient consumers.
        update_consumer_inputs(original_grads, merged_grads)
        if zero_v0():
          # If zero v0 is enabled, make sure ALLREDUCE happens before
          # Zero apply_gradients and Broadcast.
          # TODO(sayang): zero-related ops do not need to depend on all grads.
          update_ops = tfops.get_collection("zero_update")
          for op in update_ops:
            op._add_control_inputs([t.op for t in merged_grads])

  def offload_weight(self):
    """Offload weight to CPU."""
    cpu = Env.get().cluster.current_worker_cpu()
    ops = list(self._graph.operations.values())
    for op in ops:
      # For variable, apply, save/restore, set the device to CPU.
      if op.phase in [ModelPhase.APPLY, ModelPhase.SAVE_AND_RESTORE] \
          or self._graph.is_vars_related(op):
        op.set_device(cpu)
      # Replace the variable read with new identity op.
      # Control the read execution,
      # let it depends on other inputs of compute op.
      if op.phase in [ModelPhase.FORWARD, ModelPhase.BACKWARD]:
        for idx, inp in enumerate(op.inputs):
          if inp.op.type == 'Identity' and \
              common.is_variable(inp.op.inputs[0].op):
            with tfops.name_scope(constant.OFFLOAD_SCOPE_NAME):
              var_ts = inp.op.inputs[0]
              new_read = array_ops.identity(var_ts.primitive_obj,
                                            name=var_ts.op.name)
            op.update_input(idx, new_read)
            other_inp = [p.op for i, p in enumerate(op.primitive_obj.inputs)
                         if i != idx]
            if other_inp and not new_read.op.control_inputs:
              new_read.op._add_control_inputs(other_inp)

  def serialize(self):
    return "".join("{Graph:{%s}, Env:{%s}}" % (self._graph, Env.get()))

  def __str__(self):
    return self.serialize()

  def __repr__(self):
    return self.serialize()


def gcd(x, y):
  """Calculate greatest common divisor of x and y."""
  if not y:
    return x
  if x % y == 0:
    return y
  return gcd(y, x % y)


def get_global_gcd_from_dict(original_dict):
  """Get global gcd for all values in original_dict."""
  if not original_dict:
    return 1
  values_list = []
  for key in original_dict:
    values_list.append(original_dict[key])
  num_values = len(values_list)
  if num_values == 1:
    return values_list[0]
  global_gcd = gcd(values_list[0], values_list[1])
  for index in range(2, num_values):
    global_gcd = gcd(global_gcd, values_list[index])
  return global_gcd

def fetch_slice_objects_proportion_to_local_num_replicas(
    rank,
    object_list,
    global_num_replicas,
    all_devices,
    drop_last=False,
    unbalanced_slicing=False,
    name="objects"):
  """Slice objects proportion to local_num_replicas."""
  obj_list = copy.deepcopy(object_list)
  slice_objects = []
  rank_to_local_num_replicas = dict()
  for dev in all_devices:
    task_idx = common.get_task_index_from_device_str(dev)
    if task_idx in rank_to_local_num_replicas:
      rank_to_local_num_replicas[task_idx] += 1
    else:
      rank_to_local_num_replicas[task_idx] = 1
  global_gcd = get_global_gcd_from_dict(rank_to_local_num_replicas)
  object_num = len(obj_list)
  object_num_division = global_num_replicas // global_gcd
  object_slicing_enabled = True

  def fetch_final_obj_list(final_obj_list):
    """Copy data multi-times to balance load across constructors."""
    balanced_obj_num_multiple = \
        rank_to_local_num_replicas[rank] // global_gcd \
        if rank in rank_to_local_num_replicas else 1
    if balanced_obj_num_multiple > 1:
      final_obj_list *= balanced_obj_num_multiple
      random.shuffle(final_obj_list)
    tf_logging.warn("Extend {} {} times to balance load for "
                    "this constructor with num_local_replicas {}." \
                    .format(name, balanced_obj_num_multiple, \
                    rank_to_local_num_replicas[rank]))
    return final_obj_list

  if not object_num // object_num_division:
    # Disable IO-Slicing, copy object to balance all constructors.
    object_slicing_enabled = False
    slice_objects = fetch_final_obj_list(obj_list)
    tf_logging.warn("Disable io slicing balance for no enough files, "
                    "and ignore env on io-slicing.")
  else:
    remainder = object_num % object_num_division
    if remainder:
      if not unbalanced_slicing:
        if drop_last:
          obj_list = obj_list[:-remainder]
        else:
          # Disable IO-Slicing, copy object to balance all constructors.
          object_slicing_enabled = False
          slice_objects = fetch_final_obj_list(obj_list)
  if object_slicing_enabled:
    start_replica_index = 0
    end_replica_index = 0
    rank_to_local_num_replicas_keys = sorted(rank_to_local_num_replicas)
    for key in rank_to_local_num_replicas_keys:
      if key <= rank:
        start_replica_index = end_replica_index
        end_replica_index += (rank_to_local_num_replicas[key] // global_gcd)
    for obj_idx, obj_name in enumerate(obj_list):
      balanced_replica_index = obj_idx % object_num_division
      if balanced_replica_index >= start_replica_index and \
          balanced_replica_index < end_replica_index:
        slice_objects.append(obj_name)

  return slice_objects
