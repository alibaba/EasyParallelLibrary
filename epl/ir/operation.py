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
"""Implementation of graph operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from epl.ir.sharding_base import ShardingBase
from epl.ir.phase import ModelPhase
from epl.ir.tensor import Tensor
from epl.utils import constant


class Operation(ShardingBase):
  """A operation is a node of epl graph contains computation and its
  taskgraph info."""
  def __init__(self, primitive_obj, taskgraph, phase, graph):
    self._primitive_obj = primitive_obj
    self._phase = phase
    self._type = primitive_obj.type
    self._taskgraph = taskgraph
    self._graph = graph
    self._gradient_ops = []
    self._outputs = [Tensor(out, self) for out in primitive_obj.outputs]
    self._get_inputs()
    self._device_already_set = False
    self._function = None
    self.is_vars_related = None
    self._control_inputs_consumers = []
    self.add_control_inputs_consumers()

    super(Operation, self).__init__(None, None, primitive_obj.name)

  def _get_inputs(self):
    """Get operation inputs."""
    inputs = []
    get_fn = self._graph.get_function_tensor_by_name \
        if self._phase == ModelPhase.ADD_FUNCTION \
        else self._graph.get_tensor_by_name
    for inp in self._primitive_obj.inputs:
      if inp.name.startswith(constant.PARALLEL_STRATEGY):
        continue
      tensor = get_fn(inp.name)
      # TODO(wangang.ang): duplicated consumer may be
      # added to consumer_list of tensor, should be fixed
      # when multi-consumer is supported for gradient aggregation
      tensor.add_consumer(self)
      inputs.append(tensor)
    return inputs

  @property
  def phase(self):
    return self._phase

  @property
  def type(self):
    return self._type

  @property
  def device(self):
    return self._primitive_obj.device

  @property
  def node_def(self):
    return self._primitive_obj.node_def

  @property
  def op_def(self):
    return self._primitive_obj.op_def

  @property
  def output_types(self):
    return self._primitive_obj._output_types  # pylint: disable=protected-access

  @property
  def input_types(self):
    return self._primitive_obj._input_types  # pylint: disable=protected-access

  @property
  def inputs(self):
    return self._get_inputs()

  @property
  def outputs(self):
    return self._outputs

  @property
  def control_inputs(self):
    return self._primitive_obj.control_inputs

  @control_inputs.setter
  def control_inputs(self, value):
    value = copy.copy(value)
    self._primitive_obj._remove_all_control_inputs()  # pylint: disable=protected-access
    self._primitive_obj._add_control_inputs(value)  # pylint: disable=protected-access

  @property
  def control_inputs_consumers(self):
    return self._control_inputs_consumers

  def add_control_inputs_consumers(self):
    """Operations that treat self as control input."""
    for c_inp_op in self.control_inputs:
      c_inp_op = self._graph.get_operation_by_name(c_inp_op.name)
      # Cases for function operation.
      if not c_inp_op:
        continue
      c_inp_op.control_inputs_consumers.append(self)

  def add_control_inputs(self, dep_ops):
    primitive_dep_ops = [dep_op.primitive_obj for dep_op in dep_ops]
    self._primitive_obj._add_control_inputs(primitive_dep_ops)  # pylint: disable=protected-access

  @property
  def function(self):
    if self._function not in self._graph.functions:
      return None
    return self._graph.functions[self._function]

  @property
  def taskgraph(self):
    return self._taskgraph

  @taskgraph.setter
  def taskgraph(self, taskgraph):
    self._taskgraph = taskgraph

  @property
  def graph(self):
    return self._taskgraph.graph

  @property
  def gradient_ops(self):
    return self._gradient_ops

  @property
  def primitive_obj(self):
    return self._primitive_obj

  def get_attr(self, name):
    """Get attribute by name."""
    return self._primitive_obj.get_attr(name)

  def set_attr(self, name, value):
    """Set attribute by name."""
    return self._primitive_obj._set_attr(name, value)  # pylint: disable=protected-access

  def add_gradient_op(self, gradient_op):
    self._device_already_set = True
    self._gradient_ops.append(gradient_op)

  def set_device(self, device):
    self._primitive_obj._set_device(device)  # pylint: disable=protected-access

  @property
  def device_already_set(self):
    return self._device_already_set

  def update_input(self, index, tensor):
    if isinstance(tensor, Tensor):
      tensor = tensor.primitive_obj
    self._primitive_obj._update_input(index, tensor)  # pylint: disable=protected-access

  def get_control_flow_context(self):
    return self._primitive_obj._get_control_flow_context()  # pylint: disable=protected-access

  def set_control_flow_context(self, context):
    self._primitive_obj._set_control_flow_context(context)  # pylint: disable=protected-access

  def set_function(self, function):
    self._function = function

  def __str__(self):
    return \
        "epl.Operation(name='%s', device=%s, type=%s, function=%s, phase=%s)" \
        % (self._primitive_obj.name, self.device, self._type,
           self._function, self.phase)

  def __repr__(self):
    return self.__str__()
