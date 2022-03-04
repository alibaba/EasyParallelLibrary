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
"""Implementation of distributed math function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import deprecation

from epl.env import Env
from epl.ops import bridging_layer
from epl.parallel.ops import create_simple_communicator
from epl.utils import common
from epl.utils import constant


class DistributedArgmax(object):
  """Distributed version of argmax."""
  def __init__(self,
               logits,
               axis,
               output_type,
               start_dim,
               shard_index,
               all_devices,
               name="distributed_argmax"):
    self.logits = logits
    self.axis = axis
    self.output_type = output_type
    self.start_dim = start_dim
    self.shard_index = shard_index
    self.all_devices = all_devices
    self.name = name

  def call(self):
    with ops.name_scope(self.name), \
        ops.device(self.all_devices[self.shard_index]):
      predictions = self._dist_argmax()
      return predictions

  def _dist_argmax(self):
    """Kernel implementation of distributed softmax cross entropy."""
    shard_number = len(self.all_devices)

    local_max_ids = math_ops.argmax(
        self.logits, self.axis, output_type=self.output_type) + self.start_dim
    local_max_logits = math_ops.reduce_max(self.logits, self.axis)

    local_max_ids = array_ops.expand_dims(local_max_ids, self.axis)
    local_max_logits = array_ops.expand_dims(local_max_logits, self.axis)

    comm_logit = create_simple_communicator(name="MAX_LOGITS_CONCAT_FOR_ACC",
                                            devices=self.all_devices)
    global_max_logits = comm_logit.allgather(local_max_logits)
    comm_id = create_simple_communicator(name="MAX_LOGITS_ID_CONCAT_FOR_ACC",
                                         devices=self.all_devices)
    global_max_ids = comm_id.allgather(local_max_ids)
    global_max_ids = math_ops.to_float(global_max_ids)

    global_max_logits = array_ops.split(global_max_logits, shard_number, 0)
    global_max_logits = array_ops.concat(global_max_logits, axis=1)

    global_max_ids = array_ops.split(global_max_ids, shard_number, 0)
    global_max_ids = array_ops.concat(global_max_ids, axis=1)

    global_max_second_level_ids = math_ops.argmax(global_max_logits,
                                                  axis=self.axis,
                                                  output_type=self.output_type)

    max_index_mask = array_ops.one_hot(global_max_second_level_ids,
                                       global_max_ids.shape[1])

    global_max_ids = math_ops.reduce_max(global_max_ids * max_index_mask,
                                         axis=self.axis)

    if self.output_type == dtypes.int64:
      return math_ops.to_int64(global_max_ids)
    return math_ops.to_int32(global_max_ids)


def distributed_argmax(inputs,
                       axis=None,
                       name=None,
                       dimension=None,
                       output_type=dtypes.int64):
  """Functional interface for the distributed argmax."""
  axis = deprecation.deprecated_argument_lookup("axis", axis, "dimension",
                                                dimension)
  if axis is None:
    axis = 0
  env = Env.get()
  context = env.strategy_context
  split_strategy = context.split_strategy
  if split_strategy is None:
    raise RuntimeError("Got none split strategy from context.")
  strategy_devices = split_strategy.devices
  for index, device in enumerate(strategy_devices):
    if common.get_task_index_from_device_str(device) == env.cluster.worker_index:
      with ops.device(device), \
          ops.name_scope(name, "distributed_argmax"):
        start_dim = env.parallel_information[constant.INFO_KEY_START_DIM][index]
        layer = DistributedArgmax(inputs, axis, output_type, start_dim, index,
                                  strategy_devices)
        return layer.call()
  raise RuntimeError("This is not a construct worker.")


def distributed_equal(predictions, labels, name):
  """Functional interface for the distributed equal."""
  if labels is None:
    raise ValueError("Labels must not be None.")
  if predictions is None:
    raise ValueError("Logits must not be None.")

  env = Env.get()
  context = env.strategy_context
  split_strategy = context.split_strategy
  if split_strategy is None:
    raise RuntimeError("Got none split strategy from context.")
  strategy_devices = split_strategy.devices
  for device in strategy_devices:
    if common.get_task_index_from_device_str(device) == env.cluster.worker_index:
      with ops.device(device), \
          ops.name_scope(name, "distributed_equal"):
        labels = bridging_layer.Replica2Split("LABEL_INPUTS_FOR_ACC")(labels)
        predictions.get_shape().assert_is_compatible_with(labels.get_shape())
        if labels.dtype != predictions.dtype:
          predictions = math_ops.cast(predictions, labels.dtype)
        is_correct = math_ops.equal(labels, predictions, name)
        return is_correct
  raise RuntimeError("This is not a construct worker.")
