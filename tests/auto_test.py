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
"""Test for auto parallel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.env import Env


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

  def test_auto_pipeline(self):
    config = epl.Config()
    config.auto.auto_parallel = True
    config.pipeline.num_micro_batch = 8
    config.pipeline.num_stages = 4
    epl.init(config)
    with tf.Graph().as_default():
      train_opts, hooks = self._model_def()

      with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        graph = epl.Graph.get()
        self.assertEqual(Env.get().config.pipeline.num_micro_batch, 8)
        self.assertEqual(Env.get().config.pipeline.num_stages, 4)
        self.assertEqual(graph.num_stages, 4)
        self.assertEqual(graph.taskgraphs[0].num_replicas, 1)
        while not sess.should_stop():
          train_loss, _, step = sess.run(train_opts)
          print("Iteration %s , Loss: %s ." % (step, train_loss))
        print("Train Finished.")

  def _auto_pipeline_repeated(self, num_layers):
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    for i in range(num_layers):
      with tf.name_scope('s1/s2/s3/s4_layer{}'.format(i)):
        for j in range(5):
          x = tf.layers.dense(inputs=x, units=16, use_bias=False, activation='relu')
          x = tf.clip_by_norm(x, 2.0)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    with tf.train.MonitoredTrainingSession() as sess:
      taskgraphs = epl.Graph.get().taskgraphs
      self.assertEqual(len(taskgraphs), 4)
      stage2layers = {}
      for tg in taskgraphs:
        operations = tg.operations.forward_operations(0, 0)
        names = [o.name for o in operations]
        layer_names = set([name.split('/')[3].split('_')[1] for name in names if 'layer' in name])
        stage2layers[tg.index] = layer_names
      sess.run(train_op)
    return stage2layers, taskgraphs

  def test_auto_pipeline_repeated(self):
    config = epl.Config({"pipeline.num_micro_batch": 4,
                         "pipeline.num_stages": 4,
                         "auto.auto_parallel": True})
    epl.init(config)
    stage2layers, taskgraphs = self._auto_pipeline_repeated(8)
    self.assertEqual(len(stage2layers), 4)
    self.assertEqual(stage2layers, {0: {'layer0', 'layer1'}, 1: {'layer3', 'layer2'}, 2: {'layer5', 'layer4'}, 3: {'layer6', 'layer7'}})
    for i in range(4):
      device = '/job:worker/replica:0/task:0/device:GPU:{}'.format(i)
      self.assertEqual(taskgraphs[i].virtual_device.get_device(0, 0), device)
      self.assertEqual(taskgraphs[i].operations.forward_operations(0, 0)[-1].device, device)

  def test_auto_pipeline_repeated2(self):
    config = epl.Config({"pipeline.num_micro_batch": 4,
                         "pipeline.num_stages": 4,
                         "auto.auto_parallel": True})
    epl.init(config)
    stage2layers, taskgraphs = self._auto_pipeline_repeated(6)
    self.assertEqual(len(stage2layers), 4)
    self.assertEqual(stage2layers, {0: {'layer0', 'layer1'}, 1: {'layer1', 'layer2', 'layer3'}, 2: {'layer3', 'layer4'}, 3: {'layer4', 'layer5'}})
    for i in range(4):
      device = '/job:worker/replica:0/task:0/device:GPU:{}'.format(i)
      self.assertEqual(taskgraphs[i].virtual_device.get_device(0, 0), device)
      self.assertEqual(taskgraphs[i].operations.forward_operations(0, 0)[-1].device, device)

  def test_auto_pipeline_repeated3(self):
    config = epl.Config({"pipeline.num_micro_batch": 4,
                         "pipeline.num_stages": 4,
                         "auto.auto_parallel": True})
    epl.init(config)
    stage2layers, taskgraphs = self._auto_pipeline_repeated(12)
    self.assertEqual(len(stage2layers), 4)
    self.assertEqual(stage2layers, {0: {'layer0', 'layer1', 'layer2'},
                                    1: {'layer3', 'layer4', 'layer5'},
                                    2: {'layer6', 'layer7', 'layer8'},
                                    3: {'layer9', 'layer10', 'layer11'}})
    for i in range(4):
      device = '/job:worker/replica:0/task:0/device:GPU:{}'.format(i)
      self.assertEqual(taskgraphs[i].virtual_device.get_device(0, 0), device)
      self.assertEqual(taskgraphs[i].operations.forward_operations(0, 0)[-1].device, device)


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  test.main()
