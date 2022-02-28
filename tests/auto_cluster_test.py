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
"""Test for cluster."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test
import epl
from epl.env import Env
from epl.ir.graph import Graph
from epl.ir.phase import ModelPhase


# pylint: disable=missing-docstring,unused-argument,unused-variable,
# pylint: disable=protected-access
class AutoClusterTest(test.TestCase):

  def test_pipeline(self):
    config = epl.Config()
    config.pipeline.num_micro_batch = 2
    epl.init(config)
    with tf.Graph().as_default():
      with epl.replicate(name="stage_0", device_count=1):
        num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
        num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
            .batch(10).repeat(1)
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                             iterator.initializer)
        x, _ = iterator.get_next()
        dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
      with epl.replicate(name="stage_1", device_count=1):
        logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
        loss = tf.reduce_mean(logits)
      g = Graph.get()
      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)
      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      tvars = tf.trainable_variables()
      grads = tf.gradients(loss, tvars)
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
      train_op = optimizer.apply_gradients(list(zip(grads, tvars)))
      self.assertTrue(Env.get().cluster is not None)
      self.assertEqual(Env.get().cluster.worker_num, 1)
      self.assertEqual(len(Env.get().cluster.virtual_devices), 2)
      self.assertEqual(Env.get().cluster.virtual_devices[0]._slice_devices,
                       [['/job:worker/replica:0/task:0/device:GPU:0'], ['/job:worker/replica:0/task:0/device:GPU:2']])
      self.assertEqual(Env.get().cluster.virtual_devices[1]._slice_devices,
                       [['/job:worker/replica:0/task:0/device:GPU:1'], ['/job:worker/replica:0/task:0/device:GPU:3']])
      with tf.train.MonitoredTrainingSession() as sess:
        loss_value = sess.run([train_op, loss])
        print(loss_value)

  def test_model_parallelism(self):
    epl.init()
    with tf.Graph().as_default():
      with epl.replicate(name="stage_0", device_count=1):
        num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
        num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
            .batch(10).repeat(1)
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                             iterator.initializer)
        x, _ = iterator.get_next()
        dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
      with epl.replicate(name="stage_1", device_count=1):
        logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
        loss = tf.reduce_mean(logits)
      g = Graph.get()
      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)
      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      tvars = tf.trainable_variables()
      grads = tf.gradients(loss, tvars)
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
      train_op = optimizer.apply_gradients(list(zip(grads, tvars)))
      self.assertTrue(Env.get().cluster is not None)
      self.assertEqual(Env.get().cluster.worker_num, 1)
      self.assertEqual(len(Env.get().cluster.virtual_devices), 2)
      self.assertEqual(Env.get().cluster.virtual_devices[0]._slice_devices,
                       [['/job:worker/replica:0/task:0/device:GPU:0'], ['/job:worker/replica:0/task:0/device:GPU:2']])
      self.assertEqual(Env.get().cluster.virtual_devices[1]._slice_devices,
                       [['/job:worker/replica:0/task:0/device:GPU:1'], ['/job:worker/replica:0/task:0/device:GPU:3']])
      with tf.train.MonitoredTrainingSession() as sess:
        loss_value = sess.run([train_op, loss])
        print(loss_value)

  def test_dp(self):
    epl.init()
    with tf.Graph().as_default():
      with epl.replicate(name="replica", device_count=1):
        num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
        num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
        dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
            .batch(10).repeat(1)
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                             iterator.initializer)
        x, _ = iterator.get_next()
        dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
        logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
        loss = tf.reduce_mean(logits)
      g = Graph.get()
      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)
      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      tvars = tf.trainable_variables()
      grads = tf.gradients(loss, tvars)
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
      train_op = optimizer.apply_gradients(list(zip(grads, tvars)))
      self.assertTrue(Env.get().cluster is not None)
      self.assertEqual(Env.get().cluster.worker_num, 1)
      self.assertEqual(len(Env.get().cluster.virtual_devices), 1)
      self.assertEqual(Env.get().cluster.virtual_devices[0]._slice_devices,
                       [['/job:worker/replica:0/task:0/device:GPU:0'], ['/job:worker/replica:0/task:0/device:GPU:1'],
                        ['/job:worker/replica:0/task:0/device:GPU:2'], ['/job:worker/replica:0/task:0/device:GPU:3']])
      with tf.train.MonitoredTrainingSession() as sess:
        loss_value = sess.run([train_op, loss])
        print(loss_value)

# pylint: enable=missing-docstring,unused-argument,unused-variable,
# pylint: enable=protected-access

if __name__ == '__main__':
  test.main()
