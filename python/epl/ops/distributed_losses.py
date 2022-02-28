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
"""Implementation of distributed losses function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import losses

from epl.env import Env
from epl.parallel.ops import create_simple_communicator
from epl.ops import bridging_layer
from epl.utils import common
from epl.utils import constant


class DistributedSoftmaxCrossEntropy(object):
  """Distributed version of softmax cross entropy loss."""
  def __init__(self,
               labels,
               logits,
               start_dim,
               shard_index,
               all_devices,
               name="distributed_softmax_cross_entropy"):
    self.labels = labels
    self.logits = logits
    self.start_dim = start_dim
    self.shard_index = shard_index
    self.all_devices = all_devices
    self.name = name

  def call(self):
    with ops.name_scope(self.name), \
        ops.device(self.all_devices[self.shard_index]):
      loss = self._dist_softmax_cross_entropy_kernel()
      return loss

  def _dist_softmax_cross_entropy_kernel(self):
    """Kernel implementation of distributed softmax cross entropy."""
    shard_number = len(self.all_devices)

    local_max_logits = array_ops.expand_dims(math_ops.reduce_max(self.logits,
                                                                 axis=1),
                                             axis=1)
    comm = create_simple_communicator(name="MAX_LOGITS_CONCAT",
                                      devices=self.all_devices)
    global_max_logits = comm.allgather(local_max_logits)
    global_max_logits = array_ops.split(global_max_logits, shard_number, 0)
    global_max_logits = array_ops.concat(global_max_logits, axis=1)
    global_max_logits = math_ops.reduce_max(global_max_logits, axis=1)

    ones = array_ops.ones(shape=self.logits.shape[1])
    ones = array_ops.expand_dims(ones, axis=1)
    max_logits_expand = array_ops.expand_dims(global_max_logits, axis=1)
    max_logits_expand = math_ops.matmul(max_logits_expand,
                                        ones,
                                        transpose_b=True)
    shift_logits = self.logits - max_logits_expand

    logits_exp = math_ops.exp(shift_logits)

    local_sum = array_ops.expand_dims(math_ops.reduce_sum(logits_exp, axis=1),
                                      axis=1)
    comm = create_simple_communicator(name="SUM_NORMALIZERS",
                                      devices=self.all_devices)
    global_sum = comm.batch_allreduce(local_sum)
    prob_log = shift_logits - math_ops.log(global_sum + 1e-8)
    mask = self._get_mask()
    loss = math_ops.reduce_sum(prob_log * mask) / math_ops.negative(
        (math_ops.cast(array_ops.shape(prob_log)[0], dtypes.float32)))

    return loss

  def _get_mask(self):
    """Get mask information to choose corresponding logits."""
    shard_begin = self.start_dim
    shard_end = self.start_dim + self.logits.shape[1]
    shard_lmask = math_ops.cast(
        gen_math_ops.greater_equal(self.labels, shard_begin), dtypes.float32)
    shard_rmask = math_ops.cast(gen_math_ops.less(self.labels, shard_end),
                                dtypes.float32)
    shard_mask = shard_lmask * shard_rmask
    shard_mask = gen_array_ops.reshape(shard_mask, [-1, 1])
    shift_labels = self.labels - shard_begin
    mask = array_ops.one_hot(shift_labels, self.logits.shape[1])
    mask = math_ops.multiply(mask, shard_mask)
    mask = gen_array_ops.stop_gradient(mask)

    return mask


def distributed_sparse_softmax_cross_entropy_with_logits(
    labels,
    logits,
    weights=1.0,
    strategy=None,
    loss_collection=ops.GraphKeys.LOSSES,
    reduction=losses.losses_impl.Reduction.SUM_BY_NONZERO_WEIGHTS):
  """Functional interface for the distributed sparse softmax cross
  entropy with logits."""
  if labels is None:
    raise ValueError("Labels must not be None.")
  if logits is None:
    raise ValueError("Logits must not be None.")

  env = Env.get()
  context = env.strategy_context
  split_strategy = context.split_strategy
  if split_strategy is None:
    raise RuntimeError("Got none split strategy from context.")
  strategy_devices = split_strategy.devices
  for index, device in enumerate(strategy_devices):
    if common.get_task_index_from_device_str(device) == Env.get().cluster.worker_index:
      with ops.device(device), \
          ops.name_scope(strategy, "dist_softmax_cross_entropy_with_logits"):
        labels = bridging_layer.Replica2Split("LABEL_INPUTS")(labels)
        start_dim = env.parallel_information[constant.INFO_KEY_START_DIM][index]
        layer = DistributedSoftmaxCrossEntropy(labels, logits, start_dim,
                                               index, strategy_devices)

        local_loss = layer.call()
        if weights is not None:
          local_loss = losses.losses_impl.compute_weighted_loss(
              local_loss, weights, strategy,
              loss_collection, reduction=reduction)
        else:
          losses.util.add_loss(local_loss, loss_collection)

        comm = create_simple_communicator(name="LOSS_REDUCE",
                                          devices=strategy_devices)
        final_loss = comm.batch_allreduce(local_loss)
        return final_loss
  raise RuntimeError("This is not a construct worker.")
