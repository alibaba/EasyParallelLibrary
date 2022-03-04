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
# ==============================================================================
"""Implementation of graph Defun."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from epl.utils import constant
from epl.env import Env


class DefinedFunction(object):
  """A DefinedFunction is produced by function defined."""
  def __init__(self, name, taskgraph):
    self._device = None
    self._name = name
    self._nodes = dict()
    self._taskgraph = taskgraph
    self._is_dataset_related = False

  @property
  def name(self):
    return self._name

  @property
  def device(self):
    return self._device

  @property
  def is_dataset_related(self):
    return self._is_dataset_related

  @is_dataset_related.setter
  def is_dataset_related(self, is_dataset_related):
    self._is_dataset_related = is_dataset_related

  @property
  def primitive_obj(self):
    return ops.get_default_graph()._functions[self.name]  # pylint: disable=protected-access

  @property
  def func(self):
    return self.primitive_obj._func  # pylint: disable=protected-access

  @property
  def input_types(self):
    return self.primitive_obj._input_types  # pylint: disable=protected-access

  @property
  def arg_names(self):
    return self.primitive_obj._arg_names  # pylint: disable=protected-access

  @property
  def grad_func(self):
    return self.primitive_obj._grad_func  # pylint: disable=protected-access

  @property
  def python_grad_func(self):
    return self.primitive_obj._python_grad_func  # pylint: disable=protected-access

  @property
  def out_names(self):
    return self.primitive_obj._out_names  # pylint: disable=protected-access

  @property
  def shape_func(self):
    return self.primitive_obj._shape_func  # pylint: disable=protected-access

  @property
  def capture_by_value(self):
    return self.primitive_obj._capture_by_value  # pylint: disable=protected-access

  @property
  def whitelisted_stateful_ops(self):
    return self.primitive_obj._whitelisted_stateful_ops  # pylint: disable=protected-access

  @property
  def capture_resource_var_by_value(self):
    return self.primitive_obj._capture_resource_var_by_value  # pylint: disable=protected-access

  @property
  def extra_kwargs(self):
    return self.primitive_obj._extra_kwargs  # pylint: disable=protected-access

  @property
  def taskgraph(self):
    return self._taskgraph

  @taskgraph.setter
  def taskgraph(self, taskgraph):
    # If auto parallel is enabled, do not raise Exception
    # since auto parallel may partition taskgraphs
    if not Env.get().config.auto.auto_parallel and self._taskgraph is not None and \
       self._taskgraph.index != taskgraph.index:
      raise RuntimeError(
          "Taskgraph is already set for function {}. while {}/{}".format(
              self.name, self.taskgraph.index, taskgraph.index))
    self._taskgraph = taskgraph

  def get_all_nodes(self):
    """Get all nodes in self."""
    if self._nodes:
      return self._nodes
    for function_def in \
        ops.get_default_graph().as_graph_def().library.function:
      if function_def.signature.name.startswith(constant.PARALLEL_STRATEGY):
        continue
      if function_def.signature.name == self.name:
        for node in function_def.node_def:
          node = self._taskgraph.graph.get_function_operation_by_name(node.name)
          if not node:
            continue
          self.add_node(node)
          node.set_function(self.name)
          self._device = node.device

    return self._nodes

  def get_node_by_name(self, name):
    if name not in self.nodes:
      raise RuntimeError("Node {} not in {}.".format(name, self))
    return self.nodes[name]

  def add_node(self, node):
    self._device = node.device
    self._nodes[node.name] = node

  @property
  def nodes(self):
    if self._nodes:
      for node in list(self._nodes.values()):
        node.taskgraph = self.taskgraph
    return self._nodes.values() \
        if self._nodes else self.get_all_nodes()

  def __str__(self):
    return "epl.DefinedFunction(name=%s, is_dataset_related=%s, nodes=%s)" \
        % (self.name, self.is_dataset_related, self.nodes)

  def __repr__(self):
    return self.__str__()
