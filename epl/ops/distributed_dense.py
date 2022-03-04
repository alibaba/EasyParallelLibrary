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
"""Implementation of distributed dense layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs

from epl.env import Env
from epl.ops import bridging_layer
from epl.ops import initializers
from epl.utils import common
from epl.utils import constant


class DistributedDense(object):
  """Distributed fully connected layer for model parallelism.

  Args:
    inputs: Part of inputs of distributed layer.
    units: Positive integer, dimensionality of the output space.
    shard_index: Device of current part of fully connected layer.
    all_devices: All devices to place fully connected layer.
    activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    kernel_constraint: Constraint function applied to
        the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such cases.
  """
  def __init__(self,
               inputs,
               units,
               shard_index,
               all_devices,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=init_ops.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None):
    self.inputs = inputs
    self.in_shape = inputs.shape
    self.units = int(units)
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer,
                                               fan_in=self.in_shape[-1].value,
                                               fan_out=self.units)
    self.bias_initializer = initializers.get(bias_initializer,
                                             fan_out=self.units)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.trainable = trainable
    self.name = name
    self.shard_index = shard_index
    self.all_devices = all_devices
    if self.shard_index > len(self.all_devices):
      raise ValueError("Shard index is larger than all devices count."
                       "Shard Index: %s, All devices count: %s" %
                       (self.shard_index, len(self.all_devices)))

  def build(self):
    """Construct dense layer."""

    part_column = int(self.units / len(self.all_devices))
    remainder = self.units % len(self.all_devices)
    start_dim = 0
    if self.shard_index == 0:
      part_column = part_column + remainder
    else:
      start_dim = part_column * self.shard_index + remainder
    self._set_start_dim(start_dim)

    self.kernel = vs.get_variable(name="kernel_%s" % self.shard_index,
                                  shape=[self.in_shape[-1].value, part_column],
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint,
                                  trainable=self.trainable)
    if self.use_bias:
      self.bias = vs.get_variable(name="bias_%s" % self.shard_index,
                                  shape=[part_column],
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  trainable=self.trainable)
    else:
      self.bias = None

  def call(self):
    """Do dense layer calculation."""
    with ops.name_scope("%s_%s" % (self.name, self.shard_index)), \
        ops.device(self.all_devices[self.shard_index]):
      self.build()
      outputs = math_ops.matmul(self.inputs, self.kernel)
      if self.use_bias:
        outputs = math_ops.add(outputs, self.bias)
      if self.activation is not None:
        outputs = self.activation(outputs)
    return outputs

  def _set_start_dim(self, start_dim):
    parallel_info = Env.get().parallel_information
    if constant.INFO_KEY_START_DIM not in parallel_info:
      parallel_info[constant.INFO_KEY_START_DIM] = [0] * len(self.all_devices)
    parallel_info[constant.INFO_KEY_START_DIM][self.shard_index] = start_dim


def distributed_dense(inputs,
                      units,
                      activation=None,
                      use_bias=True,
                      kernel_initializer=None,
                      bias_initializer=init_ops.zeros_initializer(),
                      kernel_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      kernel_constraint=None,
                      bias_constraint=None,
                      trainable=True,
                      name=None,
                      reuse=None):
  """Functional interface for the distributed densely-connected layer."""
  if activity_regularizer:
    raise ValueError("Activity regularizer is not supported "
                     "for distributed dense layer.")
  if reuse:
    raise ValueError("Reuse can't be true for distributed dense layer.")

  context = Env.get().strategy_context
  split_strategy = context.split_strategy
  if split_strategy is None:
    raise RuntimeError("Got none split strategy from context.")
  strategy_devices = split_strategy.devices

  for index, device in enumerate(strategy_devices):
    if common.get_task_index_from_device_str(device) == Env.get().cluster.worker_index:
      with vs.variable_scope(name, default_name="distributed_dense"), \
          ops.device(device):
        total_inputs = bridging_layer.Replica2Split("FEATURE_INPUTS")(inputs)
        layer = DistributedDense(inputs=total_inputs,
                                 units=units,
                                 shard_index=index,
                                 all_devices=strategy_devices,
                                 activation=activation,
                                 use_bias=use_bias,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 kernel_constraint=kernel_constraint,
                                 bias_constraint=bias_constraint,
                                 trainable=trainable,
                                 name=name)
        return layer.call()
  raise RuntimeError("This is not a construct worker.")
