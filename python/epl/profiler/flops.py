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
"""Flop statistics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.python.profiler.internal import flops_registry
from tensorflow.python.profiler import profiler
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training import session_run_hook


# pylint: disable=protected-access

class RegisterFlops(ops.RegisterStatistics):
  """A decorator for registering the flops for an op type.
  If it is already registered in TF, use the version in TF instead of error.
  """

  def __call__(self, f):
    """Registers "f" as the statistics function for "op_type"."""
    name = self._op_type + "," + self._statistic_type
    if name not in ops._stats_registry._registry:
      ops._stats_registry.register(f, name)
    return f


@RegisterFlops("BatchMatMul", "flops")
@RegisterFlops("BatchMatMulV2", "flops")
def _calc_batch_mat_mul_flops(graph, node):
  """Calculates the compute resources needed for BatchMatMul."""
  transpose_a = node.attr["transpose_a"].b
  a_shape = graph_util.tensor_shape_from_node_def_name(graph, node.input[0])
  a_shape.assert_is_fully_defined()
  if transpose_a:
    k = int(a_shape[-2])
  else:
    k = int(a_shape[-1])
  output_shape = graph_util.tensor_shape_from_node_def_name(graph, node.name)
  output_shape.assert_is_fully_defined()
  output_count = np.prod(output_shape.as_list())
  return ops.OpStats("flops", (k * output_count * 2))


@RegisterFlops("CumSum", "flops")
def _calc_cum_sum_flops(graph, node):
  """Compute flops for CumSum operation."""
  return flops_registry._reduction_op_flops(graph, node, reduce_flops=1,
                                            finalize_flops=0)


@RegisterFlops("Prod", "flops")
def _calc_prod_flops(graph, node):
  """Compute flops for Prod operation"""
  return flops_registry._reduction_op_flops(graph, node, reduce_flops=1,
                                            finalize_flops=0)


@RegisterFlops("Relu", "flops")
def _calc_relu_flops(graph, node):
  """Compute flops for Relu operation"""
  return flops_registry._binary_per_element_op_flops(graph, node)


@RegisterFlops("AddV2", "flops")
def _calc_add_v2_flops(graph, node):
  """Compute flops for AddV2 operation"""
  return flops_registry._add_flops(graph, node)


@RegisterFlops("FloorMod", "flops")
def _cal_floor_mod_flops(graph, node):
  """Compute flops for FloorMod operation."""
  return flops_registry._binary_per_element_op_flops(graph, node)


@RegisterFlops("FloorDiv", "flops")
def _cal_floor_div_flops(graph, node):
  """Compute flops for FloorDiv operation."""
  return flops_registry._binary_per_element_op_flops(graph, node)


@RegisterFlops("Exp", "flops")
def _cal_exp_flops(graph, node):
  """Compute flops for Exp operation."""
  return flops_registry._binary_per_element_op_flops(graph, node)


@RegisterFlops("Sqrt", "flops")
def _cal_sqrt_flops(graph, node):
  """Compute flops for Sqrt operation."""
  return flops_registry._binary_per_element_op_flops(graph, node)


@RegisterFlops("LogSoftMax", "flops")
def _cal_logsoftmax_flops(graph, node):
  # log 1, softmax 5
  return flops_registry._unary_op_flops(graph, node, ops_per_element=6)


def profile_flops(sess, run_metadata, cmds):
  """Profile flops for graph with session run."""
  opt = profiler.ProfileOptionBuilder.float_operation()
  total_flops = 0
  for cmd in cmds:
    flops = profiler.profile(sess.graph, run_meta=run_metadata,
                             cmd=cmd, options=opt)
    total_flops = flops.total_float_ops
    tf_logging.info('cmd: {},  Number of FLOPs: {}G' \
                    .format(cmd, flops.total_float_ops / 1e9))
  return total_flops


class FlopsProfilerHook(session_run_hook.SessionRunHook):
  """Captures flops profiling information at first run."""
  def __init__(self, cmds="scope"):
    super(FlopsProfilerHook, self).__init__()
    self.done = False
    if isinstance(cmds, str):
      cmds = [cmds]
    if not isinstance(cmds, list):
      raise ValueError("cmds must be str or list.")
    for cmd in cmds:
      if cmd not in ["scope", "op", "graph"]:
        raise ValueError("cmd must among [scope, op, graph], got {}" \
                         .format(cmds))
    self.cmds = cmds

  def before_run(self, run_context): # pylint: disable=unused-argument
    opts = (config_pb2.RunOptions(
        trace_level=config_pb2.RunOptions.FULL_TRACE)
            if not self.done else None)

    return SessionRunArgs({}, options=opts)

  def after_run(self, run_context, run_values):
    if not self.done:
      profile_flops(run_context.session, run_values.run_metadata, self.cmds)
      self.done = True

# pylint: enable=protected-access
