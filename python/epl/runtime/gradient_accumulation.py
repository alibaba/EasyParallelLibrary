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
"""Classes for gradient accumulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops, dtypes
from tensorflow.python.ops.init_ops import zeros_initializer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging

from epl.utils import constant
from epl.env import Env
from epl.ir.graph import Graph
from epl.runtime.optimizer_helper import filter_none_grads, \
  apply_grad_group

def ga_iter_num():
  """Return gradient accumulation iteration number."""
  ga_iters = 1
  if Graph.get().num_stages == 1:
    ga_iters = Graph.get().get_pipeline_config().num_micro_batch
  return ga_iters


def ga_enabled():
  """Determine whether gradient accumulation is enabled."""
  return ga_iter_num() > 1


def _create_accum_slots(optimizer, var_list, name):
  """Create accumulation slots."""
  slots = []
  for v in var_list:
    s = optimizer._zeros_slot(v, "accum", name) # pylint: disable=protected-access
    slots.append(s)
  return slots

def apply_accmulation(optimizer, apply_gradients_fn,
                      slots_and_vars, slots, ngroup,
                      global_step, niter):
  """Apply ga."""

  tf_logging.info("Enable gradient accumulation, num_micro_batch: {}".format(niter))
  mean_grad_flag = True if Env.get().config.communication.gradients_reduce_method == \
                           constant.REDUCE_METHOD_MEAN else False
  data = []
  grads = []
  for g, v in slots_and_vars:
    with ops.device(g.device):
      g = array_ops.identity(g, name='acc_grads_tensor')
      if mean_grad_flag:
        g = g / niter
    grads.append(g)
    data.append((g, v))
  grads_and_vars = data
  Graph.get().gradients += grads
  update_ops = []
  apply_op = apply_grad_group(optimizer, apply_gradients_fn, grads_and_vars,
                              ngroup, global_step, "epl_apply_grad_ga")
  update_ops.append(apply_op)
  with ops.control_dependencies(update_ops):
    clear_ops = [state_ops.assign(s, array_ops.zeros_like(s)) for s in slots]
  return control_flow_ops.group(*clear_ops)


def apply_ga(optimizer,
             apply_gradients_fn,
             grads_and_vars,
             global_step=None,
             niter=1,
             ngroup=1,
             name="cond_update_grad"):
  """Apply gradients every niter by ngroup."""
  grads_and_vars = filter_none_grads(grads_and_vars)
  vs = []
  for g, v in grads_and_vars:
    assert isinstance(g, (ops.Tensor, ops.IndexedSlices)) and \
            isinstance(v, variables.Variable), \
        "GradAccumOptimizer does not work for the gradient of {}! " \
        "Types of v and g are {} and {}".format(v.op.name, type(v),
                                                type(g))
    vs.append(v)

  slots = _create_accum_slots(optimizer, vs, 'AccumGrad')
  slots_and_vars = [(s, gv[1]) for s, gv in \
                    zip(slots, grads_and_vars)]
  counter = variable_scope.get_variable("counter", shape=[],
                                        dtype=dtypes.int64,
                                        initializer=zeros_initializer(),
                                        trainable=False)

  with ops.name_scope('GradAccumOptimizer'):
    slot_ops = []
    for s, gv in zip(slots, grads_and_vars):
      g, v = gv
      slot_ops.append(s.assign_add(g))

    update_counter = state_ops.assign_add(counter, 1, name='update_counter')
    update_slot_op = control_flow_ops.group(update_counter,
                                            *slot_ops,
                                            name='update_slot')
    update_cond = math_ops.equal(math_ops.mod(update_counter, niter), 0)

    with ops.control_dependencies([update_slot_op]):
      update_grad = lambda: apply_accmulation(optimizer, apply_gradients_fn,
                                              slots_and_vars, slots, ngroup,
                                              global_step, niter)
      op = control_flow_ops.cond(update_cond,
                                 update_grad,
                                 control_flow_ops.no_op,
                                 name=name).op
  return op
