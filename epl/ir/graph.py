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
"""Impementation of epl graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from collections import deque, defaultdict
import six
from tensorflow.python.estimator.model_fn import ModeKeys
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.ops import variable_scope

from epl.env import Env
from epl.ir.function import DefinedFunction
from epl.ir.operation import Operation
from epl.ir.phase import ModelPhase
from epl.ir.taskgraph import Taskgraph
from epl.ir.tensor import Tensor
from epl.strategies.replicate import Replicate
from epl.utils import common
from epl.utils import constant


class GraphKeys(object):
  """Graph keys to collect local tensors."""

  # Key to collect local tensors that are local to this replica
  # and need to allgather across all replicas.
  GLOBAL_CONCAT_OBJECTS = "global_concat_objects"
  # Key to collect local tensors that are local to this replica
  # and need to allreduce_mean across all replicas.
  GLOBAL_MEAN_OBJECTS = "global_mean_objects"
  # Key to collect local tensors that are local to this replica
  # and need to allreduce_sum across all replicas.
  GLOBAL_SUM_OBJECTS = "global_sum_objects"
  # Key to collect local tensors that are local to this replica
  # and need to allgather for this replica.
  LOCAL_CONCAT_OBJECTS = "local_concat_objects"
  # Key to collect local tensors that are local to this replica
  # and need to allreduce_mean for this replica.
  LOCAL_MEAN_OBJECTS = "local_mean_objects"
  # Key to collect local tensors that are local to this replica
  # and need to allreduce_sum for this replica.
  LOCAL_SUM_OBJECTS = "local_sum_objects"

  ALL_COLLECTION_KEYS = [
      GLOBAL_CONCAT_OBJECTS, GLOBAL_MEAN_OBJECTS, GLOBAL_SUM_OBJECTS,
      LOCAL_CONCAT_OBJECTS, LOCAL_MEAN_OBJECTS, LOCAL_SUM_OBJECTS
  ]


class Graph(object):
  """EPL graph contains computation graph and parallelism info."""
  def __init__(self):
    self._env = Env.get()
    self._reset()

  def _reset(self):
    """Reset graph content."""

    self._taskgraphs = []
    self._operations = {}
    self._tensors = {}
    self._op_name_sequence = []
    self._map_op_name_to_sequence = {}
    self._map_context_to_taskgraph = {}
    self._map_variable_scope_to_taskgraph = {}
    # Functions
    self._functions = {}
    self._function_operations = {}
    self._function_tensors = {}
    self._op_with_function_map = {}
    self._current_function_name = None
    self._current_model_phase = ModelPhase.FORWARD
    self._current_backward_op = None
    self._current_cloned_taskgraph = None
    self._vars_related_op_names = None
    self._dataset_related_ops = None
    self._clone_dataset_related_ops = False
    # TODO(jiangle.jl): Support multi-optimizer.
    self._gradients = []
    self.primitive_init_op = None
    # TODO(jiangle.jl): Support train_and_evaluate.
    self._dataset_api_op = []
    self._collections = {}
    # Map original tensor (replicas) to merged_tensor (merged_tensor_replicas)
    # for show merged outputs after sess.run.
    self._merged_outputs_map = {}
    # Map summary tag to original tensor for tensorboard after sess.run
    self._summary_map = {}
    self.reset_cache_for_clone()
    self._control_flow_context_map = {}
    self._original_context_cache = {}
    self._user_default_taskgraph = None
    self._epl_default_taskgraph = None
    # Map group name to ops.
    self.group2ops = defaultdict(list)
    # Map op to group name.
    self.op2group = {}
    # Cache operations with phase of apply/save_and_restore
    # without taskgraph. For very rare cases, some ops with
    # const type and backward phase.
    self._unclustered_operations = []
    self._strategy_context = self._env.strategy_context
    self._is_local_resources_ready = False
    self.parallel_information = {}
    self.parallel_information[constant.ALL_COMM_RESOURCES] = []
    # TODO(jiangle.jl): Remove when auto split is ready.
    self.collective_communicator = dict()
    # Model mode, can be train, predict or eval.
    self._model_mode = None
    # Record colocation information.
    self._colocation_groups = defaultdict(list)
    # Colocate taskgraph and phase
    self.colocate_taskgraph = None
    self.colocate_phase = None
    self.colocate_device = None
    # Gradient checkpoint tensors
    self.gc_tensors = []

  @property
  def colocation_groups(self):
    return self._colocation_groups

  @property
  def op_with_function_map(self):
    return self._op_with_function_map

  @property
  def original_context_cache(self):
    return self._original_context_cache

  def reset_cache_for_clone(self):
    """Cache unready inputs or control_inputs for op being cloned."""
    self._unready_inputs_cache = {}
    self._unready_control_inputs_cache = {}

  @property
  def is_local_resources_ready(self):
    return self._is_local_resources_ready

  @is_local_resources_ready.setter
  def is_local_resources_ready(self, state):
    self._is_local_resources_ready = state

  @classmethod
  def get(cls, may_create=False):
    """Get static graph."""
    tf_graph = common.get_default_tf_graph()
    env = Env.get()
    if tf_graph not in env.graph_map and may_create:
      graph = Graph()
      env.graph_map[tf_graph] = graph
      env.epl_graphs.append(graph)
    return env.default_graph

  @property
  def unready_inputs_cache(self):
    return self._unready_inputs_cache

  @property
  def unready_control_inputs_cache(self):
    return self._unready_control_inputs_cache

  @property
  def map_op_name_to_sequence(self):
    return self._map_op_name_to_sequence

  @property
  def unclustered_operations(self):
    return self._unclustered_operations

  @property
  def sequenced_operations(self):
    return [self._operations[name] for name in self._op_name_sequence]

  @property
  def operations(self):
    return self._operations

  @property
  def taskgraphs(self):
    return self._taskgraphs

  @property
  def current_model_phase(self):
    return self._current_model_phase

  @property
  def is_constructor(self):
    if not self._env.cluster.virtual_devices:
      return True
    return len(self._env.cluster.virtual_devices[0].local_devices) > 0

  @property
  def first_constructor_rank(self):
    """The first slice of cluster slices are constructors and
    the first device in the slice is the first constructor.
    This function will return the rank of first constructor."""
    return common.get_task_index_from_device_str(self._taskgraphs[0].virtual_device.get_device(0, 0))

  @property
  def constructor_task(self):
    """Get set of rank for all constructors."""
    constructor_task = set()
    cluster = self._env.cluster
    if not cluster:
      raise RuntimeError(
          "Fetch number of constructors failed for none cluster in Env.")
    for device in cluster.virtual_devices[0].all_devices:
      constructor_task.add(common.get_task_index_from_device_str(device))
    return constructor_task

  @property
  def num_constructors(self):
    """Number of constructors defined in epl.Cluster."""
    return len(self.constructor_task)

  @property
  def gradients(self):
    return self._gradients

  @gradients.setter
  def gradients(self, gradients):
    self._gradients = gradients

  @property
  def dataset_api_op(self):
    return self._dataset_api_op

  @property
  def num_stages(self):
    return len(self.taskgraphs)

  @property
  def current_cloned_taskgraph(self):
    return self._current_cloned_taskgraph

  @current_cloned_taskgraph.setter
  def current_cloned_taskgraph(self, taskgraph):
    self._current_cloned_taskgraph = taskgraph

  @property
  def clone_dataset_related_ops(self):
    return self._clone_dataset_related_ops

  @clone_dataset_related_ops.setter
  def clone_dataset_related_ops(self, clone_dataset_related_ops):
    self._clone_dataset_related_ops = clone_dataset_related_ops

  @property
  def vars_related_op_names(self):
    return self._vars_related_op_names

  def check_and_set_cloned_dataset_need_clone(self):
    """Check and set cloned_dataset_need_clone
    if ops related to dataset need to be clone for ODPS TABLE API."""
    if self._clone_dataset_related_ops:
      return
    if self.num_constructors < 1:
      return
    if self.num_constructors > 1 and \
        self.num_constructors != self._env.cluster.worker_num:
      self.clone_dataset_related_ops = True
      return
    constructor_num_device = dict()
    for devices in self._env.cluster.virtual_devices[0]:
      for device in devices:
        constructor_task_index = \
            common.get_task_index_from_device_str(device)
        if constructor_task_index in constructor_num_device:
          constructor_num_device[constructor_task_index] += 1
        else:
          constructor_num_device[constructor_task_index] = 1
    constructor_num_device_list = list(constructor_num_device.values())
    constructor_num_device_set = set(constructor_num_device_list)
    self.clone_dataset_related_ops = len(constructor_num_device_set) != 1

  def reset(self):
    """Clear tf_default_graph and epl graph."""
    # Only reset tf graph when it is not under Graph().as_default() scope.
    if ops._default_graph_stack.is_cleared(): # pylint: disable=protected-access
      ops.reset_default_graph()
    self._reset()

  @property
  def tensors(self):
    return self._tensors

  def map_variable_scope_to_taskgraph(self, vs_name):
    if vs_name not in self._map_variable_scope_to_taskgraph and \
        self._user_default_taskgraph:
      self._map_variable_scope_to_taskgraph[vs_name] = \
          self._user_default_taskgraph

  def _add_taskgraph(self):
    """Create a new taskgraph and append it to epl graph."""
    if variable_scope.get_variable_scope().reuse:
      return
    if self._taskgraphs and not self._taskgraphs[0].strategy_context:
      self._taskgraphs[0].strategy_context = self._strategy_context
    else:
      self._taskgraphs.append(
          Taskgraph(len(self._taskgraphs), self._current_model_phase, self))
    self._map_context_to_taskgraph[self._strategy_context.identity] = \
        self._taskgraphs[-1]
    if not self._strategy_context.split_strategy:
      self._epl_default_taskgraph = self._taskgraphs[-1]

  def current_scope_as_default(self):
    if self._strategy_context.update_flag:
      self._add_taskgraph()
      self._strategy_context.update_flag = False
    self._user_default_taskgraph = self._taskgraphs[-1]

  def fetch_default_taskgraph(self):
    # User defined default taskgraph preferred.
    if self._user_default_taskgraph:
      return self._user_default_taskgraph, self._current_model_phase
    if self._epl_default_taskgraph:
      return self._epl_default_taskgraph, self._current_model_phase
    raise RuntimeError("Expected default taskgraph to be set already.")

  def _try_get_original_op_name_for_gc(self, op_name):
    pattern = "%s[_]*[0-9]*/" % constant.GC_DST_SCOPE_NAME
    result = re.match(pattern, op_name)
    if not result:
      return None
    return op_name[result.span()[1]:]

  def _get_current_taskgraph(self, primitive_op):
    """Return the current taskgraph for adding operation."""
    if not self._taskgraphs:
      self._add_taskgraph()

    if self.colocate_taskgraph and self.colocate_phase:
      return self.colocate_taskgraph, self.colocate_phase

    if self._current_model_phase == ModelPhase.BACKWARD:
      # Gradient checkpoint(GC) generates some forward
      # ops in tf.gradients. Puts them with their original op together
      # in the same taskgraph.
      name = primitive_op.name
      gc_origin_name = self._try_get_original_op_name_for_gc(name)
      if gc_origin_name and gc_origin_name in self._operations:
        return self._operations[gc_origin_name].taskgraph, ModelPhase.FORWARD

      if constant.LOSS_SCALE_SCOPE_NAME in primitive_op.name:
        stage_idx = -1
        for tensor in primitive_op.inputs:
          tensor = self.get_tensor_by_name(tensor.name)
          if common.is_const(tensor.producer) or \
              common.has_const_inputs_only(tensor.producer) or \
              tensor.producer.phase not in [ModelPhase.BACKWARD]:
            continue
          current_model_phase = tensor.producer.phase
          if primitive_op.type not in ["Pack"] and tensor.taskgraph is not None:
            stage_idx = max(tensor.taskgraph.index, stage_idx)
        if stage_idx == -1:
          for ele in primitive_op.control_inputs:
            if isinstance(ele, ops.Operation):
              op = self.get_operation_by_name(ele.name)
            elif isinstance(ele, ops.Tensor):
              op = self.get_tensor_by_name(ele.name).producer
            if common.is_const(op) or common.has_const_inputs_only(op) or op.phase not in [ModelPhase.BACKWARD]:
              continue
            taskgraph = op.taskgraph
            current_model_phase = op.phase
            if primitive_op.type not in ["Pack"] and taskgraph:
              stage_idx = max(taskgraph.index, stage_idx)
        if stage_idx != -1:
          return self._taskgraphs[stage_idx], self._current_model_phase

      if self._current_backward_op:
        return self._operations[self._current_backward_op].taskgraph, \
               self._current_model_phase
      return self.last_operation().taskgraph, self._current_model_phase

    if self._current_model_phase in \
        [ModelPhase.APPLY, ModelPhase.SAVE_AND_RESTORE]:
      return None, self._current_model_phase

    # For ops whose inputs or control inputs are in apply phase, mark them as apply phase.
    op_inputs = primitive_op.control_inputs + [i.op for i in primitive_op.inputs]
    for op in op_inputs:
      wop = self.get_operation_by_name(op.name)
      if wop and wop.phase == ModelPhase.APPLY:
        return None, ModelPhase.APPLY

    if self._current_model_phase == ModelPhase.FORWARD:
      current_model_phase = self._current_model_phase
      if primitive_op.control_inputs or primitive_op.inputs:
        stage_idx = -1
        for tensor in primitive_op.inputs:
          tensor = self.get_tensor_by_name(tensor.name)
          if common.is_const(tensor.producer) or tensor.producer.phase not in [ModelPhase.BACKWARD]:
            continue
          current_model_phase = tensor.producer.phase
          if primitive_op.type not in ["Pack"] and tensor.taskgraph is not None:
            stage_idx = max(tensor.taskgraph.index, stage_idx)
            continue
        if stage_idx == -1:
          for ele in primitive_op.control_inputs:
            if isinstance(ele, ops.Operation):
              op = self.get_operation_by_name(ele.name)
            elif isinstance(ele, ops.Tensor):
              op = self.get_tensor_by_name(ele.name).producer
            if common.is_const(op) or op.phase not in [ModelPhase.BACKWARD]:
              continue
            taskgraph = op.taskgraph
            current_model_phase = op.phase
            if primitive_op.type not in ["Pack"] and taskgraph:
              stage_idx = max(taskgraph.index, stage_idx)
        if current_model_phase in [ModelPhase.BACKWARD]:
          stage_idx = 0 if stage_idx == -1 else stage_idx
          primitive_op._set_device(  # pylint: disable=protected-access
              self._taskgraphs[stage_idx].virtual_device.local_devices[
                  common.get_replica_index_from_node_name(primitive_op.name)])
          return self._taskgraphs[stage_idx], current_model_phase
      if current_model_phase in [ModelPhase.FORWARD]:
        if self._strategy_context and self._strategy_context.state[0].is_default:
          return self.fetch_default_taskgraph()

    # For variable scope with reuse = True.
    vs = variable_scope.get_variable_scope()
    vs_name = vs.name
    if vs.reuse:
      # Use context identity to get previous taskgraph.
      if self._strategy_context.identity in self._map_context_to_taskgraph:
        return self._map_context_to_taskgraph[self._strategy_context.identity], \
               self._current_model_phase
      # Return variable scope related taskgraph for user default scope.
      if vs_name in self._map_variable_scope_to_taskgraph:
        return self._map_variable_scope_to_taskgraph[vs_name], \
               self._current_model_phase
      raise RuntimeError("Can't get taskgraph of current variable scope."
                         "Variable scope name: %s" % vs.name)

    if not self._strategy_context:
      # User defined default taskgraph preferred.
      return self.fetch_default_taskgraph()
    return self._taskgraphs[-1], self._current_model_phase

  def _add_function_to_taskgraph(self, op, taskgraph):
    """Add function to graph if op has attribute function."""
    op.set_function(self._current_function_name)
    if self._current_function_name and \
        self._current_function_name not in self._functions:
      self._functions[self._current_function_name] = \
          DefinedFunction(
              self._current_function_name, \
              None)

    if self._current_function_name:
      self._functions[self._current_function_name].add_node(op)

    for attr_name in constant.FUNCTION_TYPE:
      attr = op.node_def.attr.get(attr_name)
      if attr is None:
        continue
      if attr.func is None:
        continue
      self._op_with_function_map[op] = attr_name
      if self._current_function_name and \
          (common.get_replica_index_from_node_name(
              str(self._current_function_name)) or \
           common.get_micro_batch_index_from_node_name(
               str(self._current_function_name))):
        continue
      if attr.func.name not in self._functions:
        defun = DefinedFunction(str(attr.func.name), taskgraph)
        self._functions[str(defun.name)] = defun
      else:
        defun = self._functions[str(attr.func.name)]
      defun.taskgraph = taskgraph
      if taskgraph is not None:
        taskgraph.add_functions(defun)

  @property
  def current_function_name(self):
    return self._current_function_name

  @current_function_name.setter
  def current_function_name(self, func_name):
    self._current_function_name = func_name

  @property
  def functions(self):
    return self._functions

  @property
  def strategy_context(self):
    return self._strategy_context

  def add_operation(self, primitive_op):
    """Add operation to graph."""
    if not self.is_constructor:
      return

    for colocate_name in primitive_op.colocation_groups():
      self._colocation_groups[colocate_name].append(primitive_op)
    if primitive_op.name.startswith(constant.PARALLEL_STRATEGY):
      op = Operation(primitive_op, None, self._current_model_phase, self)
      self._operations[str(op.name)] = op
      for t in op.outputs:
        self._tensors[t.name] = t
      return

    if self._strategy_context.update_flag:
      self._add_taskgraph()
      self._strategy_context.update_flag = False

    current_taskgraph, current_model_phase = \
        (self.current_cloned_taskgraph, self._current_model_phase) \
        if self.current_cloned_taskgraph is not None \
        else self._get_current_taskgraph(primitive_op)
    op = Operation(primitive_op, current_taskgraph, current_model_phase, self)
    if self.colocate_device:
      op.set_device(self.colocate_device)
    # Specific dataset api op.
    if op.type in constant.DATASET_API_OPS:
      self._dataset_api_op.append(op)
      if op.type in constant.ODPS_TABLE_API_OPS:
        self.check_and_set_cloned_dataset_need_clone()

    if current_model_phase == ModelPhase.ADD_FUNCTION:
      if not self._current_function_name:
        self._function_operations[op.name] = op
        for t in op.outputs:
          self._function_tensors[t.name] = t
        op.set_function(self.current_function_name)
    else:
      for t in op.outputs:
        self._tensors[t.name] = t
      self._operations[str(op.name)] = op
      self._map_op_name_to_sequence[str(op.name)] = len(self._op_name_sequence)
      self._op_name_sequence.append(op.name)
      if current_taskgraph is not None:
        self.link_operation_to_taskgraph(op, current_taskgraph)
      else:
        self._unclustered_operations.append(op)
      if self._current_model_phase == ModelPhase.BACKWARD and \
          self._current_backward_op in self._operations:
        self._operations[self._current_backward_op].add_gradient_op(op)
    self._add_function_to_taskgraph(op, current_taskgraph)

    # Reset current cloned taskgraph.
    self.current_cloned_taskgraph = None

  def link_operation_to_taskgraph(self, op, taskgraph):
    """Put operations into related taskgraph."""
    taskgraph.add_operation(op, op.phase)
    taskgraph.add_tensors(op.outputs)
    op.taskgraph = taskgraph

  def serialize(self):
    return "".join("{TaskgraphIndex:%s, TaskgraphContent:{%s}}" % (i, taskgraph)
                   for i, taskgraph in enumerate(self._taskgraphs))

  def __str__(self):
    return self.serialize()

  def __repr__(self):
    return self.serialize()

  def format(self, max_depth=-1, prefix_list=None):
    """
    Format Taskgraph(Stage) with indentation and device information.
    :param max_depth: the max depth of scope is used,
           e.g. if Scope name is "epl/stage0/conv/kernel" and max_depth is 3,
           then "epl/stage0/conv/" is kept.
    :param prefix_list: if prefix_list is not empty, all op name that
           starts with prefix in the prefix_list is used
    """
    return "\n\n".join(
        taskgraph.format(max_depth, prefix_list)
        for taskgraph in self._taskgraphs)

  def add_to_collection(self, tensors, graph_key):
    """Add tensors to collection by graph_key."""
    if not self.is_constructor:
      return
    if not self._env.is_ready:
      tf_logging.warn(
          "epl.add_to_collection is ignored for env uninitialized.")
      return
    if not self._is_graph_key_valid(graph_key):
      raise KeyError("%s not supported in GraphKeys of epl" % graph_key)
    if not isinstance(tensors, list):
      tensors = [tensors]
    for idx, tensor in enumerate(tensors):
      if not isinstance(tensor, ops.Tensor):
        raise ValueError("Type %s not supported in epl collections."
                         "Expected tf.tensor." % tensor)
      tensors[idx] = self.get_tensor_by_name(tensor.name)
    if graph_key not in self._collections:
      self._collections[graph_key] = tensors
    else:
      self._collections[graph_key].extend(tensors)

  def _is_graph_key_valid(self, graph_key):
    return graph_key in GraphKeys.ALL_COLLECTION_KEYS

  def get_collection(self, graph_key):
    """Fetch objects list in collection with graph_key."""
    # TODO(jiangle.jl): Support user-defined key.
    if not self.is_constructor:
      return []
    if not self._env.is_ready:
      tf_logging.warn(
          "epl.get_collection is ignored for env uninitialized.")
      return []
    if not self._is_graph_key_valid(graph_key):
      raise KeyError("%s not in GraphKeys of epl" % graph_key)
    return self._collections[graph_key] \
        if graph_key in self._collections else []

  def get_all_collections(self):
    """Fetch all collections."""
    if not self.is_constructor:
      return []
    if not self._env.is_ready:
      tf_logging.warn(
          "epl.get_all_collections is ignored for env uninitialized.")
      return []
    return [self.get_collection(graph_key) \
        for graph_key in GraphKeys.ALL_COLLECTION_KEYS] \
        if self.is_constructor else []

  @property
  def merged_outputs_map(self):
    return self._merged_outputs_map

  @property
  def summary_map(self):
    return self._summary_map

  @property
  def control_flow_context_map(self):
    return self._control_flow_context_map

  @control_flow_context_map.setter
  def control_flow_context_map(self, value):
    self._control_flow_context_map = value

  def _name_validity_check(self, name, object_type):
    if not isinstance(name, six.string_types):
      raise TypeError("%s names are strings, not %s." %
                      (object_type, type(name).__name__))

  def get_function_by_name(self, name):
    self._name_validity_check(name, "Function")
    if name not in self._functions:
      raise ValueError("Function does not exist. Function" " name: %s" % name)
    return self._functions[name]

  def get_operation_by_name(self, name):
    self._name_validity_check(name, "Operation")
    if name not in self._operations:
      return None
    return self._operations[name]

  def get_function_operation_by_name(self, name):
    self._name_validity_check(name, "Function Operation")
    if name in self._function_operations:
      return self._function_operations[name]
    return None

  def get_tensor_by_name(self, name):
    self._name_validity_check(name, "Tensor")
    if name not in self._tensors:
      return None
    return self._tensors[name]

  def get_function_tensor_by_name(self, name):
    self._name_validity_check(name, "Function Tensor")
    if name not in self._function_tensors:
      raise ValueError("Function tensor does not exist. "
                       "Function tensor name: %s." % name)
    return self._function_tensors[name]

  def set_model_phase(self, phase):
    self._current_model_phase = phase

  def last_operation(self):
    return self._operations[self._op_name_sequence[-1]]

  def set_current_backward_op(self, op):
    self._current_backward_op = op.name

  def traverse_depend_ops(self, seed_ops, consider_outputs=False):
    """Traverse the graph to get all depend operations of seed_ops. All depend
    operations of each seed_op will be saved in one list as return result.

    Args:
      seed_ops: Indicating the seed operations to start traversing. It
       could be a string type of operation name or operation type. List
       type of string or operation is also supported.
      consider_outputs: Boolean type. Traverse output consumers of the depended
        operations if true, else only input producers will be considered. All
        node could be traversed if with_outputs set to be true. You should
        know exactly what you're doing before setting with_oututs to be true.
    """
    if isinstance(seed_ops, str):
      seed_ops = [seed_ops]
    seed_ops = [
        self.get_operation_by_name(item) if isinstance(item, str) else item
        for item in list(seed_ops)
    ]
    depend_ops_result = []
    ops_to_check = deque(seed_ops)
    ops_already_check = set()
    seed_ops_count = len(ops_to_check)

    def _correct_control_input(c_inp, consumer_op):
      if consumer_op.phase == ModelPhase.ADD_FUNCTION:
        return self.get_function_operation_by_name(c_inp.name)
      return self.get_operation_by_name(c_inp.name)

    while ops_to_check:
      op = ops_to_check.popleft()
      ops_already_check.add(op)

      for attr_name in constant.FUNCTION_TYPE:
        attr = op.node_def.attr.get(attr_name)
        if attr is None or attr.func is None:
          continue
        if str(attr.func.name) not in self._functions:
          self._functions[str(attr.func.name)] = \
              DefinedFunction(str(attr.func.name), None)
        if self._functions[str(attr.func.name)].nodes:
          depend_ops_result += self._functions[str(attr.func.name)].nodes
          ops_to_check += self._functions[str(attr.func.name)].nodes
        self._functions[str(attr.func.name)].is_dataset_related = True

      for inp in op.inputs:
        inp = self.get_tensor_by_name(inp.name) \
            if not isinstance(inp, Tensor) else inp
        producer = inp.producer
        if producer in ops_already_check or \
            producer.type in constant.EXCLUDED_DEPEND_OPS:
          continue
        depend_ops_result.append(producer)
        ops_to_check.append(producer)

      for c_inp in op.control_inputs:
        c_inp = _correct_control_input(c_inp, op)
        if not c_inp:
          continue
        if c_inp in ops_already_check or \
            c_inp.type in constant.EXCLUDED_DEPEND_OPS:
          continue
        depend_ops_result.append(c_inp)
        ops_to_check.append(c_inp)

      seed_ops_count -= 1
      if consider_outputs and seed_ops_count < 0:
        for out in op.outputs:
          for consumer in out.consumers:
            if consumer in ops_already_check or \
                consumer.type in constant.EXCLUDED_DEPEND_OPS:
              continue
            depend_ops_result.append(consumer)
            ops_to_check.append(consumer)
    return depend_ops_result

  def get_dataset_related_ops(self):
    """Get dataset related ops except DATASET_OPS."""
    if self._dataset_related_ops is None:
      seed_ops = []
      for op in self._operations:
        op_type = self.get_operation_by_name(op).type
        if op_type in constant.DATASET_OPS:
          seed_ops.append(op)
      self._dataset_related_ops = set(self.traverse_depend_ops(seed_ops, True))
    return self._dataset_related_ops

  def need_clone(self, op):
    """Return true if operation need to be cloned, otherwise false."""
    if isinstance(op, Operation):
      if self.current_model_phase == ModelPhase.MICRO_BATCH_CLONE:
        return (op.graph.pipeline_enabled) \
            and (not self.is_vars_related(op)) \
            and (not self.is_dataset_related(op)) \
            and (not self.is_global_step_related(op))
      elif self.current_model_phase == ModelPhase.REPLICATED:
        if self.clone_dataset_related_ops:
          return op.taskgraph.local_num_replicas > 1
        return (not self.is_dataset_related(op)) and \
             op.taskgraph.local_num_replicas > 1
      else:
        raise RuntimeError('%s do not support need_clone check.' %
                           self.current_model_phase)
    else:
      raise RuntimeError("%s is not supported to clone." % op)

  def is_vars_related(self, op):
    """Check if some op related to variables defined."""
    if op.is_vars_related is not None:
      return op.is_vars_related

    if self._vars_related_op_names is None:
      self._vars_related_op_names = []
      for taskgraph in self._taskgraphs:
        for ele in taskgraph.operations.forward_operations(0, 0):
          if ele.type in ['VariableV2', 'Variable']:
            self._vars_related_op_names.append(ele.name)
            ele.is_vars_related = True

    op.is_vars_related = not all((not op.name.startswith(prefix)) \
        for prefix in self._vars_related_op_names)
    return op.is_vars_related

  @staticmethod
  def is_dataset_type(op):
    """Check if some op related to dataset ops defined."""
    return op.type in constant.DATASET_OPS

  def is_dataset_related(self, op_or_tensor):
    """Check if some op related to dataset defined except DATASET_OPS."""
    if isinstance(op_or_tensor, (ops.Tensor, Tensor)):
      if isinstance(op_or_tensor, ops.Tensor):
        op_or_tensor = self.get_tensor_by_name(op_or_tensor.name)
      op_or_tensor = op_or_tensor.producer
    elif not isinstance(op_or_tensor, (ops.Operation, Operation)):
      raise TypeError("Unsupported type check for object {} with type {}".format(op_or_tensor.name, op_or_tensor.type))
    op = self.get_operation_by_name(op_or_tensor.name)
    dataset_related_ops = self.get_dataset_related_ops()
    return op in dataset_related_ops

  @staticmethod
  def is_global_step_related(obj):
    """Check if some object related to global step ops defined."""
    return obj.name.find('global_step') >= 0

  def get_local_replicas(self, obj):
    """Get replicas of object for data parallelism.

    Args:
      obj: Operation or Tensor type to get its replicas.
    """
    replicas = []
    if obj is None:
      return replicas
    if isinstance(obj, Operation):
      get_fn = self.get_operation_by_name
    elif isinstance(obj, Tensor):
      get_fn = self.get_tensor_by_name
    else:
      raise ValueError("Object type is not supported for getting replicas. Object type: %s" % type(obj))
    local_num_replicas = obj.taskgraph.local_num_replicas
    for replica_index in range(1, local_num_replicas):
      replica_prefix = common.get_replica_prefix(replica_index)
      replicated_obj = get_fn(replica_prefix + obj.name)
      # For object without replicas.
      if replicated_obj is None:
        break
      replicas.append(replicated_obj)
    return replicas

  def get_local_micro_batches(self, obj):
    """Get micro-batch replicas of object for pipeline.

    Args:
      obj: Operation or Tensor type to get its micro-batch replicas.
    """
    if isinstance(obj, Operation):
      get_fn = self.get_operation_by_name
    elif isinstance(obj, Tensor):
      get_fn = self.get_tensor_by_name
    else:
      raise ValueError("Object type is not supported for getting micro-batch replicas. Object type: %s" % type(obj))
    num_micro_batch = obj.taskgraph.pipeline_config.num_micro_batch
    replica_prefix = common.get_replica_prefix_from_node_name(obj.name)
    micro_batches = []
    for micro_batch_idx in range(1, num_micro_batch):
      micro_batch_prefix = common.get_micro_batch_prefix(micro_batch_idx)
      prefix = replica_prefix + micro_batch_prefix
      cloned_obj = get_fn(prefix + obj.name[len(replica_prefix):])
      # For object without micro-batch replicas.
      if cloned_obj is None:
        break
      micro_batches.append(cloned_obj)
    return micro_batches

  def get_pipeline_config(self):
    """Get pipeline config for epl graph, and unique expected."""
    pipeline_config = self.taskgraphs[0].pipeline_config
    for index in range(1, len(self.taskgraphs)):
      taskgraph = self.taskgraphs[index]
      if (pipeline_config.num_micro_batch != taskgraph.pipeline_config.num_micro_batch or
          pipeline_config.strategy != taskgraph.pipeline_config.strategy):
        raise RuntimeError("Pipeline Config is not unique.")
      pipeline_config = taskgraph.pipeline_config
    return pipeline_config

  @property
  def pipeline_enabled(self):
    """Check if pipeline is enabled."""
    if self.num_stages == 1:
      return False
    return self.get_pipeline_config().num_micro_batch > 1


  @property
  def need_parallel(self):
    """Check if the graph need do parallelism."""
    # If model mode is train or not set, return True by default.
    if self._model_mode == ModeKeys.TRAIN or not self._model_mode:
      return True
    # Only support evaluate in task0.
    return False

  def set_model_mode(self, mode):
    self._model_mode = mode

  @property
  def model_mode(self):
    return self._model_mode

  def set_default_strategy(self, strategy):
    """Set default strategy."""
    if isinstance(strategy, Replicate):
      self._env.strategy_context.default_strategy = strategy
      self.current_scope_as_default()
      tf_logging.info("Set default strategy {}.".format(strategy))
    else:
      raise Exception("Only replicate is supported as default strategy now, got {} ".format(strategy))


def add_to_collection(tensor, graph_key):
  Graph.get().add_to_collection(tensor, graph_key)


def get_collection(graph_key):
  return Graph.get().get_collection(graph_key)


def get_all_collections():
  return Graph.get().get_all_collections()
