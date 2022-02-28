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
"""Test for set_default_scope."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.config import Config
from epl.env import Env
from epl.ir.graph import Graph
from epl.strategies.replicate import Replicate


# pylint: disable=missing-docstring,protected-access,unused-argument
# pylint: disable=line-too-long,bad-continuation,unused-variable

def get_slices(cluster):
  slices = []
  for vd in cluster.virtual_devices:
    slices.append(vd._slice_devices)
  return slices


class SetDefaultScopeTest(test.TestCase):
  """Test import functions of parallelism transformation"""
  def _model_def(self):
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    return tf.reduce_mean(logits)

  def test_basic_info_of_default_scope(self):
    config = epl.Config()
    config.cluster.colocate_split_and_replicate = True
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    # Check Env
    self.assertTrue(Env.get().strategy_context.state[0].is_default)
    self.assertTrue(isinstance(Env.get().strategy_context.default_strategy, Replicate))
    # Check Cluster
    self.assertTrue(len(Env.get().cluster.virtual_devices) == 1)
    self.assertEqual(get_slices(Env.get().cluster),
                    [[["/job:worker/replica:0/task:0/device:GPU:0"],
                      ["/job:worker/replica:0/task:0/device:GPU:1"]]])
    self.assertTrue(len(Graph.get().taskgraphs) == 1)
    self.assertTrue(Graph.get()._user_default_taskgraph.index == 0)

  def test_illegal_calls(self):

    with self.assertRaises(Exception):
      config = "Illeagl config"
      epl.init(config)

  def test_graph_with_clip(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    g = Graph.get()
    loss = self._model_def()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
        log_device_placement=False))

    # check taskgraph.
    self.assertTrue(len(g.taskgraphs) == 1)
    local_num_replicas = g.taskgraphs[0].local_num_replicas
    self.assertTrue(local_num_replicas == 2)
    b_exit_op_list = list(g.taskgraphs[0].backward_exit_ops(0, 0))
    b_exit_op_list = [b_exit_op.name for b_exit_op in b_exit_op_list]
    list.sort(b_exit_op_list)
    for i in range(local_num_replicas):
      prefix = "EPL_REPLICA_{}/".format(i) if i else ""
      b_exit_op_list = [prefix + ele for ele in b_exit_op_list]
      self.assertEqual(b_exit_op_list, [
          prefix + "clip_by_global_norm/clip_by_global_norm/_0",
          prefix + "clip_by_global_norm/clip_by_global_norm/_1",
          prefix + "clip_by_global_norm/clip_by_global_norm/_2",
          prefix + "clip_by_global_norm/clip_by_global_norm/_3"
      ])

  def test_graph_with_clip_and_scale(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    g = Graph.get()
    loss = self._model_def()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    # Scale gradients manually
    grads = [grad * float(1 / 2) for grad in grads]
    optimizer.apply_gradients(list(zip(grads, tvars)))
    tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
        log_device_placement=False))

    # check taskgraph.
    self.assertTrue(len(g.taskgraphs) == 1)
    local_num_replicas = g.taskgraphs[0].local_num_replicas
    self.assertTrue(local_num_replicas == 2)
    b_exit_op_list = list(g.taskgraphs[0].backward_exit_ops(0, 0))
    b_exit_op_list = [b_exit_op.name for b_exit_op in b_exit_op_list]
    list.sort(b_exit_op_list)
    for i in range(local_num_replicas):
      prefix = "EPL_REPLICA_{}/".format(i) if i else ""
      b_exit_op_list = [prefix + ele for ele in b_exit_op_list]
      self.assertEqual(b_exit_op_list, [
          prefix + "mul", prefix + "mul_1", prefix + "mul_2", prefix + "mul_3"
      ])

  def test_graph_with_clip_after_allreduce(self):
    conf = Config()
    # Clip gradients after allreduce
    conf.communication.clip_after_allreduce = True
    epl.init(conf)
    epl.set_default_strategy(epl.replicate(1))
    g = Graph.get()
    loss = self._model_def()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    optimizer.apply_gradients(list(zip(grads, tvars)))
    tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
        log_device_placement=False))

    # check taskgraph.
    self.assertTrue(len(g.taskgraphs) == 1)
    local_num_replicas = g.taskgraphs[0].local_num_replicas
    self.assertTrue(local_num_replicas == 2)
    b_exit_op_list = list(g.taskgraphs[0].backward_exit_ops(0, 0))
    b_exit_op_list = [b_exit_op.name for b_exit_op in b_exit_op_list]
    list.sort(b_exit_op_list)
    for i in range(local_num_replicas):
      prefix = "EPL_REPLICA_{}/".format(i) if i else ""
      b_exit_op_list = [prefix + ele for ele in b_exit_op_list]
      self.assertEqual(b_exit_op_list, [
          prefix + "gradients/dense/BiasAdd_grad/BiasAddGrad",
          prefix + "gradients/dense/MatMul_grad/MatMul_1",
          prefix + "gradients/dense_1/BiasAdd_grad/BiasAddGrad",
          prefix + "gradients/dense_1/MatMul_grad/MatMul_1"
      ])


# pylint: enable=missing-docstring,protected-access,unused-argument,
# pylint: enable=line-too-long,bad-continuation,unused-variable

if __name__ == "__main__":
  test.main()
