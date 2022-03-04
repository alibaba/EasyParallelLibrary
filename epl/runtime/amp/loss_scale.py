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
"""Loss scale for auto mixed precision."""

from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework.versions import __version__
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops

from epl.env import Env
from epl.parallel.ops import Colocate
from epl.runtime.amp import loss_scale_tf
from epl.utils import constant


def _loss_scaler():
  """Get loss scaler."""
  scaler = Env.get().parallel_information.get("AMP_LOSS_SCALE")
  if scaler is None:
    scaler = loss_scale_tf.get(Env.get().config.amp.loss_scale)
    Env.get().parallel_information["AMP_LOSS_SCALE"] = scaler
  return scaler


def amp_loss_scale():
  """Get loss scale."""
  scaler = _loss_scaler()
  return scaler()


def amp_update(grads_and_vars, apply_fn, name):
  scaler = _loss_scaler()
  grads = [g for g, _ in grads_and_vars]
  loss_scale_update_op, should_apply_grads = scaler.update(grads)
  maybe_apply_op = smart_cond.smart_cond(should_apply_grads, apply_fn,
                                         control_flow_ops.no_op)
  return control_flow_ops.group(maybe_apply_op,
                                loss_scale_update_op, name=name)


def scale_loss(loss, loss_scale):
  """Scale loss by loss_scale."""
  if callable(loss):
    return lambda: loss() * loss_scale
  return loss * loss_scale

def _scale_grad(grad, loss_scale_reciprocal):
  """Scale grad value by loss_scale_reciprocal."""
  def _scale(tensor, scale_value):
    with Colocate(tensor):
      return math_ops.mul(tensor, scale_value, name=tensor.op.name + constant.EPL_AMP_LOSS_SCALE_SUFFIX)

  if isinstance(grad, ops.IndexedSlices):
    grad_vals = _scale(grad.values, loss_scale_reciprocal)
    return ops.IndexedSlices(grad_vals, grad.indices, grad.dense_shape)
  return _scale(grad, loss_scale_reciprocal)


def unscale_grads(grads, loss_scale):
  """Unscale grads by loss_scale."""
  loss_scale_reciprocal = 1 / loss_scale
  unscaled_grads = []
  for g in grads:
    if g is None:
      unscaled_grads.append(None)
    else:
      with Colocate(g):
        unscaled_grads.append(_scale_grad(g, loss_scale_reciprocal))
  return unscaled_grads
