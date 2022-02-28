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
"""Zero for data parallel partition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging

from epl.env import Env
from epl.ir.graph import Graph
from epl.parallel.ops import create_serial_communicator
from epl.runtime.optimizer_helper import apply_grad_group, \
  group_list, seperate_grads_and_vars_by_replica, filter_none_grads
from epl.utils import constant


def get_zero_level():
  """Get and check zero level."""
  global_zero_level = Env.get().parallel_information.get("ZERO_LEVEL")
  if global_zero_level:
    return global_zero_level
  zero_level = Env.get().config.zero.level
  if not zero_level:
    global_zero_level = "zero_disable"
  elif isinstance(zero_level, str) and zero_level.lower() in ["v0", "v1"]:
    global_zero_level = zero_level.lower()
  else:
    tf_logging.warn("zero_level only supports level v0 or v1 by now, got {}"
                    .format(zero_level))
    global_zero_level = "zero_disable"
  Env.get().parallel_information["ZERO_LEVEL"] = global_zero_level
  return global_zero_level

def zero_enabled():
  """Check if zero is enabled."""
  global_zero_enabled = Env.get().parallel_information.get("ZERO_ENABLED")
  if global_zero_enabled is not None:
    return global_zero_enabled
  zero_level = get_zero_level()
  global_zero_enabled = True
  if zero_level not in ["v0", "v1"]:
    global_zero_enabled = False
  elif Env.get().cluster.worker_num <= 1:
    tf_logging.warn("Relica worker number must > 1 to enable zero {}." \
                    .format(zero_level))
    global_zero_enabled = False
  elif Env.get().cluster.gpu_num_per_worker != 1:
    tf_logging.warn("EPL zero {} only support Nx1 GPUs by now." \
                    .format(zero_level))
    global_zero_enabled = False
  else:
    replica_taskgraphs = [sg for sg in Graph.get().taskgraphs
                          if sg.strategy_context.replicate_strategy
                          is not None]
    if len(replica_taskgraphs) != 1:
      tf_logging.warn("EPL zero only support one replica taskgraph, got {}." \
                      .format(len(replica_taskgraphs)))
      global_zero_enabled = False
  Env.get().parallel_information["ZERO_ENABLED"] = global_zero_enabled
  return global_zero_enabled

def zero_v0():
  """Check if zero v0 is enabled."""
  return zero_enabled() and get_zero_level() == "v0"

def zero_v1():
  """Check if zero v1 is enabled."""
  return zero_enabled() and get_zero_level() == "v1"

# pylint: disable=protected-access
def apply_zero(optimizer, apply_gradients_fn, grads_and_vars, global_step,
               niter, ngroup, name):
  """Apply zero-like partition."""
  if Env.get().parallel_information.get("NUM_ZERO_PARALLEL"):
    raise RuntimeError("Zero does not supports calling " \
                       "optimizer.apply_gradients multiple times.")

  grads_and_vars = filter_none_grads(grads_and_vars)
  num_zero_parallel = Env.get().cluster.worker_num
  Env.get().parallel_information["NUM_ZERO_PARALLEL"] = num_zero_parallel
  zero_level = get_zero_level()
  tf_logging.info("Enable Zero {}".format(zero_level))

  replica_grads_and_vars, non_replica_grads_and_vars = \
      seperate_grads_and_vars_by_replica(grads_and_vars, "Zero")
  apply_opts = []
  optimizer_finish_fn = optimizer._finish
  if replica_grads_and_vars:
    partitioned = group_list(replica_grads_and_vars, num_zero_parallel)
    if len(partitioned) != num_zero_parallel:
      raise ValueError("Zero: partitioned size {} is not equal to \
                       num_zero_parallel {}" \
                       .format(len(partitioned), num_zero_parallel))
    last_bcast_op = None
    last_reduce_op = None
    worker_index = Env.get().cluster.worker_index
    taskgraph = Graph.get().get_tensor_by_name(replica_grads_and_vars[0][0] \
                                              .name).taskgraph
    bcast_comm = create_serial_communicator("BROADCAST_zero", \
                                          taskgraph.virtual_device.all_devices)
    if zero_v1():
      reduce_comm = create_serial_communicator("REDUCE_zero", \
                                             taskgraph.virtual_device.all_devices)
    # If non_replica_grads_and_vars is not empty, update global_step when
    # applying non_replica_grads_and_vars.
    # make sure global step is updated only once
    global_step_replica = global_step
    if non_replica_grads_and_vars:
      optimizer._finish = super(optimizer.__class__, optimizer)._finish
      global_step_replica = None

    for root_worker in range(len(partitioned) - 1, -1, -1):
      sub_grads_and_vars = partitioned[root_worker]
      grads = [g for g, v in sub_grads_and_vars]
      bcast_variables = [v for g, v in sub_grads_and_vars]
      bcast_deps = []
      if zero_v0():
        bcast_deps += grads
      if zero_v1():
        if last_reduce_op is not None:
          with ops.control_dependencies([last_reduce_op]):
            grads = reduce_gradients(grads, root_worker, worker_index, \
                                     reduce_comm, taskgraph)
        else:
          grads = reduce_gradients(grads, root_worker, worker_index, \
                                   reduce_comm, taskgraph)
        sub_grads_and_vars = list(zip(grads, bcast_variables))
        last_reduce_op = control_flow_ops.group([g.op for g in grads])
        bcast_deps.append(last_reduce_op)

      if root_worker == worker_index:
        if niter > 1:
          tf_logging.warn("Ignore ga with zero by now, \
                          will support in the future.")
          niter = 1
        apply_op = apply_grad_group(optimizer, apply_gradients_fn,
                                    sub_grads_and_vars, ngroup,
                                    global_step_replica,
                                    name=name)
        ops.add_to_collection("zero_update", apply_op)
        bcast_deps.append(apply_op)
        apply_opts.append(apply_op)

      if last_bcast_op:
        bcast_deps.append(last_bcast_op)
      with ops.control_dependencies(bcast_deps):
        bcast_op = broadcast_variables(bcast_variables, root_worker, \
                                       bcast_comm, taskgraph)
      last_bcast_op = bcast_op
      apply_opts.append(bcast_op)
  if non_replica_grads_and_vars:
    optimizer._finish = optimizer_finish_fn
    apply_non_replica = apply_grad_group(optimizer, apply_gradients_fn,
                                         non_replica_grads_and_vars,
                                         ngroup, global_step,
                                         name=name)
    apply_opts.append(apply_non_replica)
  return control_flow_ops.group(apply_opts)


def reduce_gradients(gradients, root_worker, worker_index, comm, taskgraph):
  """Reduce gradients to root_worker."""
  if not gradients:
    return gradients
  mean_grad_flag = True if Env.get().config.communication.gradients_reduce_method == \
                           constant.REDUCE_METHOD_MEAN else False

  with ops.device(taskgraph.virtual_device.local_devices[0]):
    reduced_gradients = comm.reduce(gradients, root_worker)
    if mean_grad_flag and root_worker == worker_index:
      replica_num = len(taskgraph.virtual_device.all_devices)
      reduced_gradients = [g / replica_num for g in reduced_gradients]
  return reduced_gradients


def broadcast_variables(bcast_variables, rank, comm, taskgraph):
  """Broadcast weight values of trainable variables."""
  assign_ops = []
  with ops.device(taskgraph.virtual_device.local_devices[0]):
    reduced_variables = comm.broadcast(bcast_variables, rank)
    if zero_v0():
      for v in reduced_variables:
        ops.add_to_collection("zero_update", v.op)
    for idx, variable in enumerate(bcast_variables):
      assign_ops.append(state_ops.assign(variable, reduced_variables[idx]))
  return control_flow_ops.group(assign_ops)
