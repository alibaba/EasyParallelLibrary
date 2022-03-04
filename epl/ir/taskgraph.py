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
"""Impementation of epl taskgraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import re
from collections import deque
from collections import defaultdict

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging

from epl.env import Env
from epl.ir.phase import ModelPhase
from epl.utils import common
from epl.utils import constant


class StageOps(object):
  """EPL operations in a EPL Taskgraph."""
  def __init__(self, taskgraph):
    self._is_apply_ops_sequenced = False
    self._is_save_and_restore_sequenced = False
    self.taskgraph = taskgraph
    nested_dict_type = lambda: defaultdict(nested_dict_type)
    self._phase2operations = nested_dict_type()

  def _get_operations_internal(self, *keys):
    """Get operations internal"""
    operations = self._phase2operations
    for idx, key in enumerate(keys):
      if idx == len(keys) - 1 and key not in operations:
        operations[key] = []
      operations = operations[key]
    return operations

  def add_operation(self, op, phase):
    """Insert op into related operations list."""
    replica_idx = common.get_replica_index_from_node_name(op.name)
    micro_batch_idx = common.get_micro_batch_index_from_node_name(op.name)
    if phase in (ModelPhase.FORWARD, ModelPhase.BACKWARD):
      operations = self._get_operations_internal(phase, replica_idx, micro_batch_idx)
    else:
      # ModelPhase.APPLY, ModelPhase.SAVE_AND_RESTORE
      operations = self._get_operations_internal(phase, replica_idx)
    operations.append(op)

  def forward_operations(self, replica_id, micro_batch_id):
    """Return forward operations list."""
    return self._get_operations_internal(ModelPhase.FORWARD, replica_id, micro_batch_id)

  def backward_operations(self, replica_id, micro_batch_id):
    """Return backward operations list."""
    return self._get_operations_internal(ModelPhase.BACKWARD, replica_id, micro_batch_id)

  def apply_operations(self, replica_id):
    """Return apply operations list."""
    operations = self._get_operations_internal(ModelPhase.APPLY, replica_id)
    if not self._is_apply_ops_sequenced:
      operations.sort(key=self._op_sequence_idx)
      self._is_apply_ops_sequenced = True
    return operations

  def save_and_restore_operations(self, replica_id):
    """Return save_and_restore operations list."""
    operations = self._get_operations_internal(ModelPhase.SAVE_AND_RESTORE, replica_id)
    if not self._is_save_and_restore_sequenced:
      operations.sort(key=self._op_sequence_idx)
      self._is_save_and_restore_sequenced = True
    return operations

  def _op_sequence_idx(self, ele):
    if ele.name in self.taskgraph.graph.map_op_name_to_sequence:
      return self.taskgraph.graph.map_op_name_to_sequence[ele.name]
    raise RuntimeError("%s not in epl graph." % ele)

  def serialize(self):
    return "epl.StageOps(forward_operations=%s, backward_operations=%s, apply_operations=%s, \
        save_and_restore_operations=%s)" \
        % (self._phase2operations.get(ModelPhase.FORWARD), self._phase2operations.get(ModelPhase.BACKWARD), \
        self._phase2operations.get(ModelPhase.APPLY), self._phase2operations.get(ModelPhase.SAVE_AND_RESTORE))

  def __str__(self):
    return self.serialize()

  def __repr__(self):
    return self.serialize()


class Taskgraph(object):
  """EPL SubGraph."""
  def __init__(self, index, phase, graph, devices=None):
    """Create a taskgraph.

    Args:
      index: Taskgraph index.
      phase: ModelPhase type represent phase of model constructing.
      graph: current epl graph.
      devices: Devices of placement.
    """
    self._strategy_context = copy.deepcopy(graph.strategy_context)
    self.graph = graph
    self._index = index
    self._phase = \
        ModelPhase.SAVE_AND_RESTORE if phase == ModelPhase.SAVE_AND_RESTORE \
        else ModelPhase.FORWARD_AND_BACKWARD_AND_APPLY
    self._virtual_device = devices
    self._operations = StageOps(self)
    self._tensors = []
    self._gradients = []
    self._functions = set()

  @property
  def virtual_device(self):
    return self._virtual_device

  def dataset_entrance_op(self, replica_idx, micro_batch_idx):
    """Use 'IteratorGetNext' as forward_entrance_ops
    for first stage if forward_entrance_ops is empty."""
    forward_entrance_ops = set()
    for op in \
        self._operations.forward_operations(replica_idx, micro_batch_idx):
      if op.type in constant.SUMMARY_TYPE or \
          op.get_control_flow_context() is not None or \
          op.type in constant.OPS_IGNORED_FOR_ENT_SCHEDULER or \
          self.graph.is_vars_related(op):
        continue
      if op.type in constant.DATASET_OPS:
        forward_entrance_ops.add(op)
      if op.type in constant.PAI_DATA_PREFETCH:
        outputs = list(op.outputs)
        for out in outputs:
          consumers = list(out.consumers)
          for consumer in consumers:
            forward_entrance_ops.add(consumer)
    return forward_entrance_ops

  def forward_entrance_ops(self, replica_idx, micro_batch_idx):
    """Get forward entrance ops which consume forward outputs
    from other taskgraphs."""
    forward_entrance_ops = set()
    if self.is_first_stage:
      forward_entrance_ops = \
          self.dataset_entrance_op(replica_idx, micro_batch_idx)
    else:
      ops_ignored_has_inputs_from_other_taskgraph = set()
      for op in \
          self._operations.forward_operations(replica_idx, micro_batch_idx):
        op_types_ignored = False
        if op.type in constant.SUMMARY_TYPE or \
            op.type in constant.OPS_IGNORED_FOR_ENT_SCHEDULER or \
            self.graph.is_vars_related(op) or \
            op.get_control_flow_context() is not None:
          op_types_ignored = True
        # Ignore ops without consumer of forward phase
        all_consumers = []
        for out in op.outputs:
          consumers = list(out.consumers)
          for consumer in consumers:
            if consumer.phase in [ModelPhase.FORWARD]:
              all_consumers.append(consumer)

        # Ignore ops without inputs from other taskgraph
        has_inputs_from_other_taskgraph = False
        # Ignore ops without inputs from self taskgraph
        has_inputs_from_self_taskgraph = False
        # Ignore ops whose all inputs are ignored
        input_valid = False
        for inp in op.inputs:
          if inp.producer.type not in constant.SUMMARY_TYPE and \
              inp.producer.type not in constant.OPS_IGNORED_FOR_ENT_SCHEDULER and \
              not self.graph.is_vars_related(inp.producer):
            input_valid = True
          if id(inp.taskgraph) != id(self):
            if inp.producer.phase == ModelPhase.FORWARD:
              has_inputs_from_other_taskgraph = True
          else:
            if inp.producer.phase == ModelPhase.FORWARD and \
                inp.producer in ops_ignored_has_inputs_from_other_taskgraph:
              has_inputs_from_other_taskgraph = True
            has_inputs_from_self_taskgraph = True

        # Execute ops which is ignored and has inputs from other taskgraph
        if (has_inputs_from_other_taskgraph and \
            not has_inputs_from_self_taskgraph) or \
            (has_inputs_from_other_taskgraph and \
            (op_types_ignored or not input_valid)):
          ops_ignored_has_inputs_from_other_taskgraph.add(op)
        if not op_types_ignored and \
            input_valid and \
            has_inputs_from_other_taskgraph and \
            has_inputs_from_self_taskgraph and \
            all_consumers:
          forward_entrance_ops.add(op)
    # Ignore ops with input from forward_entrance_ops
    forward_entrance_ops_copy = list(forward_entrance_ops)
    for op in forward_entrance_ops_copy:
      ops_to_check = deque([inp.producer for inp in list(op.inputs) \
                            if id(inp.taskgraph) == id(self)])
      ops_already_check = [op]
      while ops_to_check:
        inp_op = ops_to_check.popleft()
        if inp_op in ops_already_check:
          continue
        ops_already_check.append(inp_op)
        if inp_op in forward_entrance_ops:
          forward_entrance_ops.remove(op)
          break
        if inp_op.inputs:
          ops_to_check += [inp.producer for inp in list(inp_op.inputs) \
                           if id(inp.taskgraph) == id(self) and \
                           inp.producer not in ops_to_check]
    return forward_entrance_ops

  def forward_exit_ops(self, replica_idx, micro_batch_idx):
    """Get forward exit ops which produce forward outputs
    for next taskgraph."""
    forward_exit_ops = set()
    for op in \
        self._operations.forward_operations(replica_idx, micro_batch_idx):
      if op.type in constant.SUMMARY_TYPE or \
          op.get_control_flow_context() is not None or \
          op.type in constant.OPS_IGNORED_FOR_EXIT_SCHEDULER or \
          self.graph.is_vars_related(op) or \
          self.graph.is_dataset_related(op) or \
          constant.GC_DST_SCOPE_NAME in op.name:
        continue
      should_consider = False
      all_consumers = []
      for out in op.outputs:
        consumers = out.consumers
        if not consumers:
          continue
        all_consumers += consumers
        for consumer in consumers:
          if consumer.type in constant.SUMMARY_TYPE or \
              self.graph.is_vars_related(consumer) or \
              self.graph.is_dataset_related(consumer):
            continue
          if self.is_last_stage:
            if consumer.phase in \
                [ModelPhase.BACKWARD, \
                ModelPhase.APPLY, \
                ModelPhase.SAVE_AND_RESTORE]:
              all_consumers.append(consumer)
          else:
            if id(consumer.taskgraph) != id(self):
              all_consumers.append(consumer)
          if consumer.phase in \
              [ModelPhase.BACKWARD, \
               ModelPhase.APPLY, \
               ModelPhase.SAVE_AND_RESTORE]:
            should_consider = True
      if should_consider and all_consumers:
        forward_exit_ops.add(op)

    # Ignore ops with consumer from forward_exit_ops
    forward_exit_ops_copy = list(forward_exit_ops)
    for op in forward_exit_ops_copy:
      all_consumers = []
      for out in op.outputs:
        consumers = out.consumers
        if not consumers:
          continue
        all_consumers += \
            [consumer for consumer in consumers \
             if id(consumer.taskgraph) == id(self)]
      ops_to_check = deque(all_consumers)
      ops_already_check = [op]
      while ops_to_check:
        consumer_op = ops_to_check.popleft()
        if consumer_op in ops_already_check:
          continue
        ops_already_check.append(consumer_op)
        if consumer_op in forward_exit_ops:
          forward_exit_ops.remove(op)
          break
        for out in consumer_op.outputs:
          consumers = out.consumers
          if not consumers:
            continue
          ops_to_check += [consumer for consumer in list(consumers) \
                           if id(consumer.taskgraph) == id(self) and \
                           consumer not in ops_to_check]
    return forward_exit_ops

  def backward_entrance_ops(self, replica_idx, micro_batch_idx):
    """Get backward entrance ops which consume backward outputs
    from other taskgraphs."""
    backward_entrance_ops = set()
    ops_ignored_has_inputs_from_other_taskgraph = set()
    for op in \
         self._operations.backward_operations(replica_idx, micro_batch_idx):
      op_types_ignored = False
      if op.type in constant.SUMMARY_TYPE or \
          op.type in constant.OPS_IGNORED_FOR_ENT_SCHEDULER or \
          self.graph.is_vars_related(op) or \
          op.get_control_flow_context() is not None:
        op_types_ignored = True
      # Ignore ops without consumer of backward phase
      all_consumers = []
      for out in op.outputs:
        consumers = list(out.consumers)
        for consumer in consumers:
          if consumer.phase in [ModelPhase.BACKWARD]:
            all_consumers.append(consumer)

      # Ignore ops without inputs from other taskgraph
      has_inputs_from_other_taskgraph = False
      # Ignore ops without inputs from self taskgraph
      has_inputs_from_self_taskgraph = False
      # Ignore ops whose all inputs are ignored
      input_valid = False
      for inp in op.inputs:
        if inp.producer.type not in constant.SUMMARY_TYPE and \
            inp.producer.type not in constant.OPS_IGNORED_FOR_ENT_SCHEDULER and \
            not self.graph.is_vars_related(inp.producer):
          input_valid = True
        if self.is_last_stage:
          if inp.producer.phase == ModelPhase.FORWARD and \
              id(inp.taskgraph) == id(self):
            has_inputs_from_other_taskgraph = True
          elif inp.producer.phase == ModelPhase.BACKWARD and \
              id(inp.taskgraph) == id(self):
            has_inputs_from_self_taskgraph = True
        else:
          if id(inp.taskgraph) != id(self):
            if inp.producer.phase == ModelPhase.BACKWARD:
              has_inputs_from_other_taskgraph = True
          else:
            if inp.producer.phase == ModelPhase.BACKWARD and \
                inp.producer in ops_ignored_has_inputs_from_other_taskgraph:
              has_inputs_from_other_taskgraph = True
            has_inputs_from_self_taskgraph = True
      # Execute ops which is ignored and has inputs from other taskgraph
      if (has_inputs_from_other_taskgraph and \
          not has_inputs_from_self_taskgraph) or \
          (has_inputs_from_other_taskgraph and \
           (op_types_ignored or not input_valid)):
        ops_ignored_has_inputs_from_other_taskgraph.add(op)
      if not op_types_ignored and \
          input_valid and \
          has_inputs_from_other_taskgraph and \
          has_inputs_from_self_taskgraph and \
          all_consumers:
        backward_entrance_ops.add(op)
    # Ignore ops with input from backward_entrance_ops
    backward_entrance_ops_copy = list(backward_entrance_ops)
    for op in backward_entrance_ops_copy:
      ops_to_check = deque([inp.producer for inp in list(op.inputs) \
                            if id(inp.taskgraph) == id(self)])
      ops_already_check = [op]
      while ops_to_check:
        inp_op = ops_to_check.popleft()
        if inp_op in ops_already_check:
          continue
        ops_already_check.append(inp_op)
        if inp_op in backward_entrance_ops:
          backward_entrance_ops.remove(op)
          break
        if inp_op.inputs:
          ops_to_check += [inp.producer for inp in list(inp_op.inputs) \
                           if id(inp.taskgraph) == id(self) and \
                           inp.producer not in ops_to_check]
    return backward_entrance_ops

  def backward_exit_ops(self, replica_idx, micro_batch_idx):
    """Get backward exit ops which produce backward outputs
    for prior taskgraph."""
    backward_exit_ops = set()
    for ele in self.gradients:
      ele = self.graph.get_tensor_by_name(ele.name)
      backward_exit_ops.add(ele.producer)

    # TODO(jiangle.jl): Find backward_exit_ops from consumers.
    if backward_exit_ops:
      replica_prefix = common.get_replica_prefix(replica_idx)
      micro_batch_prefix = common.get_micro_batch_prefix(micro_batch_idx)
      backward_exit_ops = \
        [self.graph.get_operation_by_name(
            replica_prefix + micro_batch_prefix + op.name) \
         for op in list(backward_exit_ops)]
    return backward_exit_ops

  def get_variables(self, replica_idx):
    """Get variables belong to this taskgraph."""
    variables = []
    for variable in tf_variables.global_variables():
      replica_prefix = common.get_replica_prefix(replica_idx)
      var_tensor = self.graph.get_tensor_by_name(replica_prefix +
                                                 variable.name)
      if id(var_tensor.taskgraph) != id(self):
        continue
      variables.append(var_tensor.primitive_obj)
    return variables

  def add_operation(self, op, phase):
    current_op_group = Env.get().parallel_information.get("CURRENT_OP_GROUP",
                                                          None)
    if current_op_group is not None:
      self.graph.group2ops[current_op_group].append(op)
      self.graph.op2group[op.name] = current_op_group
    self._operations.add_operation(op, phase)

  def add_tensors(self, tensors):
    self._tensors += tensors

  def add_functions(self, function):
    self._functions.add(function)

  @property
  def index(self):
    return self._index

  @property
  def strategy_context(self):
    return self._strategy_context

  # todo: remove
  @property
  def context(self):
    return self._strategy_context

  @strategy_context.setter
  def strategy_context(self, context):
    self._strategy_context = copy.deepcopy(context)

  @property
  def is_first_stage(self):
    return self._index == 0

  @property
  def is_last_stage(self):
    return self._index == self.graph.num_stages - 1

  @property
  def functions(self):
    return self._functions

  @property
  def num_device_per_replica(self):
    """Get device for each taskgraph replica."""
    for strategy in self._strategy_context.state:
      if strategy.device_count is not None:
        return strategy.device_count
    return 1

  @property
  def queue_runners(self):
    """Return queue runners related to self."""
    queue_runners_list = list()
    all_queue_runners = \
        list(ops.get_collection_ref(
            ops.GraphKeys.QUEUE_RUNNERS))
    for queue in all_queue_runners:
      if id(
          self.graph.get_operation_by_name(
              queue.enqueue_ops[0].name).taskgraph) == id(self):
        queue_runners_list.append(queue)
    return queue_runners_list

  def __str__(self):
    return "epl.Taskgraph(index = %s, phase=%s, stategy_context=%s," \
           "devices=%s, functions=%s, stage_ops=%s)" %     \
           (self._index, self._phase, self._strategy_context,
            self._virtual_device, self._functions, self._operations)

  def format(self, max_depth=-1, prefix_list=None):
    """
    Format Taskgraph(Stage) with indentation and device information.
    :param max_depth: the max depth of scope is used,
           e.g. if Scope name is "epl/stage0/conv/kernel" and max_depth is 3,
           then "epl/stage0/conv/" is kept.
    :param prefix_list: if prefix_list is not empty,
           all op name that starts with prefix in the prefix_list is used
    """
    res = ""
    if prefix_list is None:
      prefix_list = []
    for rid in range(self.num_replicas):
      if rid == 1 and max_depth > 0:
        # Since EPL will add prefix before original scopes,
        # increase the max_depth by 1 if it is set.
        max_depth += 1
      title = "Taskgraph {} replica {} [Device: {}]".format(
          self.index, rid, self._virtual_device.local_devices[rid])
      res += "======= Begin {} =======".format(title)
      # Use the first micro batch.
      operations = self._operations.forward_operations(rid, 0)
      visited = set()
      for op in operations:
        # Ignore ops without inputs
        if not op.inputs:
          continue
        depth = 0
        name_scopes = op.name.split("/")
        prefix_match = re.match(r".*({}).*".format("|".join(prefix_list)),
                                op.name)
        if prefix_match is None:
          continue
        op_format = ""
        for scope in name_scopes:
          if 0 < max_depth <= depth:
            break
          depth += 1
          op_format += "/" + scope
          if op_format in visited:
            continue
          visited.add(op_format)
          res += "\n" + depth * "  " + op_format[1:]
      res += "\n======= End {} =======\n\n".format(title)
    return res.strip()

  def __repr__(self):
    return self.__str__()

  @property
  def pipeline_config(self):
    pipe = copy.deepcopy(Env.get().config.pipeline)
    if pipe.num_micro_batch <= 0:
      tf_logging.warn("Got incorrect num_micro_batch {}, set num_micro_batch to 1.".format(pipe.num_micro_batch))
      pipe.num_micro_batch = 1
    return pipe

  @property
  def local_num_replicas(self):
    """Number of replicas for local worker."""
    if self._virtual_device is None:
      raise RuntimeError("Cluster must be set before getting local_num_replicas.")
    return max(len(self._virtual_device.local_devices), 1)

  @property
  def num_replicas(self):
    """Global number of replicas."""
    if self._virtual_device is None:
      raise RuntimeError("Cluster must be set before getting num_replicas.")
    return self._virtual_device.num_replicas

  def set_device(self, virtual_device):
    self._virtual_device = virtual_device

  @property
  def phase(self):
    return self._phase

  @property
  def operations(self):
    return self._operations

  @property
  def gradients(self):
    """Get original gradients related of self."""
    if not self._gradients:
      for grad in self.graph.gradients:
        if grad is None:
          continue
        t = self.graph.get_tensor_by_name(grad.name)
        if id(t.taskgraph) == id(self):
          self._gradients.append(grad)
    return self._gradients
