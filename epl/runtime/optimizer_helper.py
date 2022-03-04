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
"""Gradient Accumulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import optimizer as tf_optimizer

from epl.ir.graph import Graph
from epl.strategies.replicate import Replicate
from epl.parallel.partitioner import partition_stages


def filter_none_grads(grads_and_vars):
  """Filter None grads."""
  return [(grads, vs) for (grads, vs) in grads_and_vars if grads is not None]


def get_size(tensor):
  """Get the element size of tensor."""
  shape = tensor.shape.as_list()
  size = 1
  for s in shape:
    if s == -1: s = 1
    size *= s
  return size


def group_list(grads_and_vars, n, balance_size=True, keep_empty=True):
  """Group data to n sublist."""
  if balance_size:
    weights = [get_size(v) for _, v in grads_and_vars]
  else:
    # balance with element num.
    weights = [1] * len(grads_and_vars)
  grouped_list = partition_stages(grads_and_vars, weights, n, \
                                  enable_logging=True)
  if not keep_empty:
    grouped_list = [g for g in grouped_list if g]
  return grouped_list


def check_optimizer(optimizer):
  """Check optimizer type."""
  try:
    if not optimizer.__class__.__bases__[0] == tf_optimizer.Optimizer:
      tf_logging.warn("Ignore group apply because optimizer type {} \
                             is not the subclass of tf Optimizer" \
                             .format(optimizer))
      return False
  except Exception as e:  # pylint: disable=broad-except
    tf_logging.warn("Ignore group apply due to error {}".format(e))
    return False
  return True


# pylint: disable=protected-access
def apply_grad_group(optimizer,
                     apply_gradients_fn,
                     grads_and_vars,
                     ngroup,
                     global_step,
                     name=""):
  """Apply gradients by groups."""
  if not name:
    name = "epl_apply_grad"
  grads_and_vars = filter_none_grads(grads_and_vars)
  if ngroup == 1 or not check_optimizer(optimizer):
    update_ops = [apply_gradients_fn(optimizer, grads_and_vars, global_step)]
  else:
    tf_logging.info("Enable group apply: apply {} groups." \
                           .format(ngroup))
    name += "_group"
    optimizer_finish = optimizer._finish

    update_ops = []

    def apply_gradients_kernel(gvs, is_last_group):
      """Kernel to group gradient updates."""
      if not gvs:
        return []
      grouped_grads_and_vars = group_list(gvs, ngroup, keep_empty=False)
      group_update_ops = []
      for idx in range(len(grouped_grads_and_vars) - 1, -1, -1):
        # Set step global_step for the last gradient apply
        # to ensure global_step update only once within one iteration.
        is_last_apply = is_last_group and idx == 0
        step = global_step if is_last_apply else None
        # Call optimizer finish only once.
        if is_last_apply:
          optimizer._finish = optimizer_finish
        else:
          optimizer._finish = super(optimizer.__class__, optimizer)._finish

        if group_update_ops:
          with ops.control_dependencies([group_update_ops[-1]]):
            update_op = apply_gradients_fn(optimizer,
                                           grouped_grads_and_vars[idx], step)
        else:
          update_op = apply_gradients_fn(optimizer,
                                         grouped_grads_and_vars[idx], step)
        group_update_ops.append(update_op)
      return group_update_ops

    scope_grads_and_vars = seperate_grads_and_vars_by_replica(grads_and_vars,
                                                              "Group Apply")
    scope_grads_and_vars = [gvs for gvs in scope_grads_and_vars if gvs]
    for i, sub_grads_and_vars in enumerate(scope_grads_and_vars):
      is_last_group = i == len(scope_grads_and_vars) - 1
      update_ops += apply_gradients_kernel(sub_grads_and_vars, is_last_group)
  return control_flow_ops.group(*update_ops, name=name)


def seperate_grads_and_vars_by_replica(grads_and_vars, prefix):
  """Seperate grads_and_vars to replica / non-replica."""
  replica_grads_and_vars = []
  non_replica_grads_and_vars = []
  for g, v in grads_and_vars:
    if in_replica_strategy(v):
      replica_grads_and_vars.append((g, v))
    else:
      non_replica_grads_and_vars.append((g, v))
  tf_logging.info(
      "{}: Replica op num: {}, None replica op num: {}" \
        .format(prefix, len(replica_grads_and_vars),
                len(non_replica_grads_and_vars)))
  return replica_grads_and_vars, non_replica_grads_and_vars


def in_replica_strategy(tensor):
  """Is tensor in replica strategy."""
  states = Graph.get().tensors[tensor.name].taskgraph.strategy_context.state
  if len(states) == 1 and isinstance(states[0], Replicate):
    return True
  return False
