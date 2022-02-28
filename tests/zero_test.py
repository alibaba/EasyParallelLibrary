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
"""Test for Zero."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.runtime.zero import zero_v0, zero_v1


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class AutoTest(test.TestCase):
  def _model_def(self):
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    max_steps = 3
    hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
    return [loss, train_op, global_step], hooks

  def _check_zero_level(self, zero_level, v0, v1):
    os.environ["TF_CONFIG"] = '{"cluster":{"worker":["127.0.0.1:8001",\
        "127.0.0.1:8002"]},"task":{"type":"worker","index":0}}'
    config = epl.Config()
    config.zero.level = zero_level
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    with tf.Graph().as_default():
      self._model_def()
      self.assertTrue(zero_v0() is v0)
      self.assertTrue(zero_v1() is v1)

  def test_zero_status(self):
    self._check_zero_level("v0", True, False)
    self._check_zero_level("V0", True, False)
    self._check_zero_level("V1", False, True)
    self._check_zero_level("v1", False, True)
    self._check_zero_level("v123", False, False)
    self._check_zero_level("", False, False)
    self._check_zero_level(None, False, False)

  def test_zero_v1(self):
    os.environ["TF_CONFIG"] = '{"cluster":{"worker":["127.0.0.1:8001","127.0.0.1:8002"]},"task":{"type":"worker","index":0}}'
    config = epl.Config()
    config.zero.level = "v1"
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    train_opts, hooks = self._model_def()
    operations = epl.Graph.get().operations.keys()
    self.assertTrue(any("BROADCAST_zero" in name for name in operations))
    self.assertTrue(any("REDUCE_zero" in name for name in operations))

  def test_zero_v0(self):
    os.environ["TF_CONFIG"] = '{"cluster":{"worker":["127.0.0.1:8001","127.0.0.1:8002"]},"task":{"type":"worker","index":0}}'
    config = epl.Config()
    config.zero.level = "v0"
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    train_opts, hooks = self._model_def()
    operations = epl.Graph.get().operations.keys()
    self.assertTrue(any("BROADCAST_zero" in name for name in operations))
    self.assertTrue(all("REDUCE_zero" not in name for name in operations))


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access


if __name__ == "__main__":
  test.main()
