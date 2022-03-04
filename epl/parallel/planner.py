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
"""Classes to generate parallelism plan."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import tf_logging
from epl.ir.graph import Graph
from epl.parallel.partitioner import partition_balance
from epl.runtime.gc.auto_gradient_checkpoint import find_repeated_blocks
from epl.utils import constant


class AutoPlanGenerator(object):
  """Generate parallel plan by automatic mechanism."""
  def __init__(self, policy):
    self._policy = policy

  def search(self):
    raise NotImplementedError


class AutoStageGenerator(object):
  """Generate stages automatically.

    Args:
      policy: auto stage policy.
      num_stage: num of stages"""
  def __init__(self, num_stages, policy=None):
    self._policy = policy
    if not self._policy:
      self._policy = constant.AUTO_STAGE_POLICY_HEURISTIC
    self._num_stages = num_stages

  def search(self, graph=None):
    """Search graph partition plan"""
    if graph is None:
      graph = Graph.get()
    assert len(graph.taskgraphs) == 1, \
      "Current only support tasks with only one taskgraph in auto mode," + " got {}".format(len(graph.taskgraphs))
    taskgraph = graph.taskgraphs[0]
    operations = taskgraph.operations.forward_operations(0, 0)
    if len(operations) <= self._num_stages:
      tf_logging.warn("operations {} less than num_stages {}".format(len(operations), self._num_stages))
    operations = self.sort_ops(operations)
    return self.partition_stages(operations)

  def sort_ops(self, ops):
    # TODO(sayang): clustering and graph topology sort
    return ops

  def partition_stages(self, operations):
    """partition stages."""
    stage_ops = None
    if self._policy == constant.AUTO_STAGE_POLICY_BALANCE_OP_NUM:
      stage_ops = self.partition_balance_op_num(operations)
    elif self._policy == constant.AUTO_STAGE_POLICY_REPEATED_LAYERS:
      stage_ops = self.partition_repeated_blocks(operations)
    elif self._policy == constant.AUTO_STAGE_POLICY_HEURISTIC or stage_ops is None:
      stage_ops = self.partition_heuristic(operations)
    for ops in stage_ops:
      if not ops:
        raise RuntimeError("stage ops is empty")
    return stage_ops

  def partition_balance_op_num(self, operations):
    """partition operations based on op num"""
    weights = [1] * len(operations)
    return partition_balance(operations, weights, self._num_stages)

  def partition_repeated_blocks(self, operations):
    """partition operations based on repeated blocks."""
    repeated_blocks = find_repeated_blocks(operations)
    op2block = {}
    for i, blocks in enumerate(repeated_blocks):
      for op in blocks:
        op2block[op] = i
    if len(repeated_blocks) < constant.MIN_REPEAT_BLOCKS or len(repeated_blocks) % self._num_stages != 0:
      return None
    block_per_stage = len(repeated_blocks) // self._num_stages
    stage_ops = [[] for _ in range(self._num_stages)]
    stage_id = 0
    for op in operations:
      current_block = op2block.get(op)
      if current_block:
        current_stage_id = current_block // block_per_stage
        stage_id = current_stage_id
        if current_stage_id > stage_id:
          stage_id = current_stage_id
      stage_ops[stage_id].append(op)
    return stage_ops

  def partition_heuristic(self, operations):
    """partition operations based on heuristic strategy."""
    stage_ops = self.partition_repeated_blocks(operations)
    if stage_ops:
      return stage_ops
    return self.partition_balance_op_num(operations)
