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
"""Select Gradient Checkpoint tensor automatically."""

import math
from collections import defaultdict
from toposort import toposort

import tensorflow.contrib.graph_editor as ge
from tensorflow.python.framework import ops
from tensorflow.contrib.graph_editor import select

from epl.utils import common
from epl.env import Env
from epl.ir.graph import Graph
from epl.ir.operation import Operation
from epl.ir.tensor import Tensor
from epl.profiler.profiler import profile_memory
from epl.utils.common import in_while_loop
from epl.utils.common import is_const
from epl.parallel.partitioner import find_repeated_blocks
from epl.parallel.partitioner import partition_balance


def toposort_ops(all_ops):
  """Topology sort of ops"""
  deps = defaultdict(set)
  for op in all_ops:
    for i in op.inputs:
      deps[op].add(i.op)
  sorted_ops = toposort(deps)
  ts_sorted_lists = []
  for operations in sorted_ops:
    # Keep ops within input all_ops
    keep = list(operations.intersection(all_ops))
    if keep:
      ts_sorted_lists.append(keep)
  return ts_sorted_lists


def is_variable_const_related(op):
  """Check if op is variable or const related."""
  op = Graph.get().get_operation_by_name(op.name)
  if Graph.get().is_vars_related(op):
    return True
  elif is_const(op):
    return True
  elif op.inputs:
    return all(is_const(inp.op) or Graph.get().is_vars_related(op) \
               for inp in op.inputs)
  return False


def filter_ops(operations):
  """Filter operations that is variable/const/dataset related."""
  return [op for op in operations if not is_variable_const_related(op) and \
          not Graph.get().is_dataset_related(op)]


def filter_tensor(tensors):
  """Filter tensors that is variable/const/dataset related."""
  return [t for t in tensors if not is_variable_const_related(t.op) and \
          not Graph.get().is_dataset_related(t.op)]


def get_entrance_exits_tensors(operations):
  """Get entrance and exit operations."""
  if not operations:
    return [], []
  if isinstance(operations[0], Operation):
    operations = [op.primitive_obj for op in operations]
  outside_input_ts, outside_output_ts, _ = \
      select.compute_boundary_ts(operations)
  entrance_ts = filter_tensor(outside_input_ts)
  exit_ts = filter_tensor(outside_output_ts)
  return entrance_ts, exit_ts


def toposort_op_group(tf_fwd_ops):
  """Sort op groups."""
  op2group = Graph.get().op2group
  deps = defaultdict(set)
  fwd_ops = Graph.get().operations.values()
  fwd_ops = [op for op in fwd_ops if not is_variable_const_related(op) \
             and op.primitive_obj in tf_fwd_ops]
  fwd_groups = set([op2group.get(op.name, op.name) for op in fwd_ops])
  for op in fwd_ops:
    group = op2group.get(op.name, op.name)
    for inp in op.inputs:
      ig = op2group.get(inp.op.name, inp.op.name)
      if ig == group: continue
      deps[group].add(ig)
  sorted_group = toposort(deps)
  res = []
  for level in sorted_group:
    level = [g for g in level if g in fwd_groups]
    if level:
      res.append(level)
  return res


def get_partition_cost(sorted_groups, tensor2bytes):
  """Get the memory cost for partition."""
  level2mem = [0] * len(sorted_groups)
  for i, groups in enumerate(sorted_groups):
    for g in groups:
      exits_ts = get_group_exit_ts(g)
      for t in exits_ts:
        level2mem[i] += tensor2bytes.get(t.name, 0)
  shrink = int((max(level2mem) - min(level2mem)) // 100)
  shrink = max(shrink, 1)
  level2mem = [m / shrink for m in level2mem]
  return level2mem


def get_ops_by_group(group_name):
  """Get ops by group name"""
  group2ops = Graph.get().group2ops
  if group_name in group2ops:
    return group2ops[group_name]
  return [Graph.get().get_operation_by_name(group_name)]


def get_group_exit_ts(group_name):
  """Get exit tensors of group."""
  return get_entrance_exits_tensors(get_ops_by_group(group_name))[1]


def search_checkpoint_balance(fwd_ops):
  """Search checkpoint tensors by balancing memory."""
  checkpoints = []
  fwd_ops = [op for op in fwd_ops if not is_variable_const_related(op)]
  tensor2bytes = profile_memory()
  sorted_groups = toposort_op_group(fwd_ops)
  level2cost = get_partition_cost(sorted_groups, tensor2bytes)
  group_cost = [(g, c) for g, c in zip(sorted_groups, level2cost) if c > 0]
  sorted_groups = [x[0] for x in group_cost]
  level2cost = [x[1] for x in group_cost]
  # Number of partitions.
  num_gc = int(math.sqrt(len(sorted_groups)))
  partitioned = partition_balance(sorted_groups, level2cost, num_gc + 1)
  for block in partitioned[:-1]:
    if not block: continue
    for g in block[-1]:
      ts = get_group_exit_ts(g)
      ts = [t for t in ts if not is_variable_const_related(t.op)]
      checkpoints.extend(ts)
  return checkpoints


def search_checkpoint_by_blocks(fwd_ops):
  """Search checkpoint by repeated blocks."""
  checkpoints = []
  candidate_blocks = find_repeated_blocks(fwd_ops)
  for block_ops in candidate_blocks:
    entrance_ts, exit_ts = get_entrance_exits_tensors(block_ops)
    exit_ts = [t for t in exit_ts if len(t.consumers()) > 0]
    checkpoints.extend(entrance_ts)
    checkpoints.extend(exit_ts)
  return checkpoints


def get_epl_taskgraph_index(tensor):
  """Get the taskgraph index of tensor."""
  return Graph.get().get_tensor_by_name(tensor.name).op.taskgraph.index


def search_checkpoint_tensors(all_tensors, ys, xs):
  """Search checkpoint tensors."""

  last_gc_taskgraph = Env.get().config.gradient_checkpoint.end_taskgraph
  fwd_ops = ops.get_default_graph().get_operations()
  fwd_ops = filter_ops(fwd_ops)
  checkpoints = search_checkpoint_by_blocks(fwd_ops)
  if not checkpoints:
    checkpoints = search_checkpoint_balance(fwd_ops)
  if last_gc_taskgraph >= 0:
    checkpoints = [t for t in checkpoints \
                   if get_epl_taskgraph_index(t) <= last_gc_taskgraph]
  if checkpoints and isinstance(checkpoints[0], Tensor):
    checkpoints = [t.primitive_obj for t in checkpoints]
  bwd_ops = fast_backward_ops(fwd_ops, ys, xs, only_differentiable=True)
  bwd_tensors = get_all_tensors(bwd_ops, ["inputs", "outputs"])
  checkpoints = list(set(checkpoints) \
                     .intersection(all_tensors) \
                     .intersection(bwd_tensors))
  return checkpoints


def get_all_tensors(operations, func_names):
  """Get all tensors for operations with given func_name."""
  if not operations:
    return []
  tensors = set()
  for func_name in func_names:
    assert hasattr(operations[0], func_name)
    assert hasattr(getattr(operations[0], func_name), "__iter__")
    for op in operations:
      for o in getattr(op, func_name):
        tensors.add(o)
  return list(tensors)


def is_consumer_in_while_loop(op):
  """Is op's consumer in while loop."""
  for t in op.outputs:
    if any(in_while_loop(c) for c in t.consumers()):
      return True
  return False


def is_differentiable(op):
  """Is op differentiable."""
  try:
    return ops._gradient_registry.lookup(op.op_def.name) is not None # pylint: disable=protected-access
  except LookupError:
    return False


def fast_backward_ops(within_ops, seed_ops, stop_at_ts,
                      only_differentiable=False):
  """Get backward pass to get ops."""
  stop_at_ts = [ts for ts in stop_at_ts if not common.is_variable(ts.op)]
  if only_differentiable:
    for op in within_ops:
      if not is_differentiable(op):
        stop_at_ts += op.outputs

  bwd_ops = set(ge.get_backward_walk_ops(seed_ops, stop_at_ts=stop_at_ts))
  stop_op = [t.op for t in stop_at_ts]
  if within_ops:
    return list(bwd_ops.intersection(within_ops).difference(stop_op))
  return list(bwd_ops.difference(stop_op))


def get_while_loop_entrance_exits(op, while_entrance_exit):
  """Get entrance and exit tensors of while loop.
  The entrance and exit are outside while loop.
  """
  while_group = Graph.get().op2group.get(op.name)
  if not while_entrance_exit[while_group]:
    entrances, exits = get_entrance_exits_tensors(
        Graph.get().group2ops[while_group])
    consumers = []
    for i in exits:
      for con in i.consumers():
        consumers.extend(con.outputs)
    while_entrance_exit[while_group] = entrances, consumers
  return while_entrance_exit[while_group]


def tf_toposort_tensors(within_ops):
  """Toposort tensors within operations."""
  while_entrance_exit = defaultdict(list)

  deps = {}
  for op in within_ops:
    if in_while_loop(op):
      entrances, exits = get_while_loop_entrance_exits(op, while_entrance_exit)
      for exit_ts in exits:
        deps[exit_ts] = deps.get(exit_ts, set())
        for entrance_ts in entrances:
          deps[exit_ts].add(entrance_ts)
    for out_ts in op.outputs:
      deps[out_ts] = deps.get(out_ts, set())
      for in_ts in op.inputs:
        if in_while_loop(in_ts.op): continue
        deps[out_ts].add(in_ts)
  sorted_ts = list(toposort(deps))
  return sorted_ts

def tf_toposort(tensors, within_ops=None, sorted_ts=None):
  """Toposort tensors."""
  if not sorted_ts:
    sorted_ts = tf_toposort_tensors(within_ops)
  # only keep the tensors from our original list
  ts_sorted_lists = []
  for ts_list in sorted_ts:
    keep = list(set(ts_list).intersection(tensors))
    if keep:
      ts_sorted_lists.append(keep)

  return ts_sorted_lists
