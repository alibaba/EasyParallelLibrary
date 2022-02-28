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
"""Test for add_weight."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion as Version
import six
import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base
from tensorflow.python.platform import test
from tensorflow.python.framework.versions import __version__

import epl
from epl.config import Config
from epl.ir.graph import Graph

new_tf_version = \
    Version(__version__) < Version("2.0") and \
    Version(__version__) >= Version("1.14.0")


# pylint: disable=missing-docstring,protected-access,unused-argument,arguments-differ
# pylint: disable=line-too-long,bad-continuation,unused-variable
class FFN(base.Layer):
  """Construct a FeedForward Networks.

    Args:
        inputs: BLM Tensor.

    Returns:
        outputs: BLM Tensor.
        aux_loss: scalar auxiliary loss.
    """
  def __init__(self, **kwargs):
    super(FFN, self).__init__(**kwargs)
    self.initializer = None
    self.num_experts = 10
    self.intermediate_size = 16
    self.hidden_size = 16
    self.activation_fn = tf.keras.activations.get("relu")

  def build(self, input_shape):

    with epl.split():
      self.in_weights = self.add_weight(shape=(self.num_experts,
                                               self.hidden_size,
                                               self.intermediate_size),
                                        initializer=self.initializer,
                                        dtype=tf.float32,
                                        name='in_weights')
      self.out_weights = self.add_weight(shape=(self.num_experts,
                                                self.intermediate_size,
                                                self.hidden_size),
                                         initializer=self.initializer,
                                         dtype=tf.float32,
                                         name='out_weights')
    super(FFN, self).build(input_shape)

  def call(self, inputs, training=True):
    with epl.split():
      assert training
      intermediate = tf.einsum('EGCM,EMH->EGCH',
                               inputs,
                               self.in_weights,
                               name="inter_outputs")
      # activation function
      activated_inters = self.activation_fn(intermediate)

      # output forward
      outputs = tf.einsum('EGCH,EHM->EGCM',
                          activated_inters,
                          self.out_weights,
                          name="outputs")
      outputs = tf.reshape(outputs, [-1, 640])
      return outputs


class AddWeightTest(test.TestCase):
  """Test import functions of parallelism transformation"""
  def _model_def(self):
    num_x = np.random.randint(0, 10, (500, 2, 20, 16)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    self.ffn = FFN()
    dense1 = self.ffn(inputs=x)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    return tf.reduce_mean(logits)

  def test_graph_with_local_clip(self):
    conf = epl.Config()
    conf.cluster.colocate_split_and_replicate = True
    epl.init(conf)
    epl.set_default_strategy(epl.replicate(1))
    g = Graph.get()
    loss = self._model_def()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads = [tf.clip_by_norm(grad, clip_norm=1.0) for grad in grads]
    optimizer.apply_gradients(list(zip(grads, tvars)))
    tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
        log_device_placement=False))

    # check taskgraph.
    self.assertTrue(len(g.taskgraphs) == 3)
    self.assertTrue(g.taskgraphs[0].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[1].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[2].local_num_replicas == 1)
    vars_list = [[], [], []]
    vars_list[0] = [
        "dense/bias/Adam:0", "dense/bias/Adam_1:0", "dense/bias:0",
        "dense/kernel/Adam:0", "dense/kernel/Adam_1:0", "dense/kernel:0"
    ]
    # TODO(jiangle.jl): Merge Taskgraph 1 and Taskgraph 2
    vars_list[1] = [
        "beta1_power:0", "beta2_power:0", "ffn/in_weights/Adam:0",
        "ffn/in_weights/Adam_1:0", "ffn/in_weights:0",
        "ffn/out_weights/Adam:0", "ffn/out_weights/Adam_1:0",
        "ffn/out_weights:0"
    ]
    vars_list[2] = []
    grads = [[], [], []]
    grads[0] = [
        "clip_by_norm_2:0",
        "clip_by_norm_3:0"
    ]
    grads[1] = []
    grads[2] = [
        "clip_by_norm:0",
        "clip_by_norm_1:0"
    ]
    for i in range(3):
      taskgraph = g.taskgraphs[i]
      var = [ele.name for ele in taskgraph.get_variables(0)]
      grd = [ele.name for ele in taskgraph.gradients]
      list.sort(var)
      list.sort(grd)
      self.assertEqual(var, vars_list[i])
      self.assertEqual(grd, grads[i])

  def test_graph_with_global_clip(self):
    conf = epl.Config()
    conf.cluster.colocate_split_and_replicate = True
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
    self.assertTrue(len(g.taskgraphs) == 3)
    self.assertTrue(g.taskgraphs[0].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[1].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[2].local_num_replicas == 1)
    vars_list = [[], [], []]
    vars_list[0] = [
        "dense/bias/Adam:0", "dense/bias/Adam_1:0", "dense/bias:0",
        "dense/kernel/Adam:0", "dense/kernel/Adam_1:0", "dense/kernel:0"
    ]
    # TODO(jiangle.jl): Merge Taskgraph 1 and Taskgraph 2
    vars_list[1] = [
        "beta1_power:0", "beta2_power:0", "ffn/in_weights/Adam:0",
        "ffn/in_weights/Adam_1:0", "ffn/in_weights:0",
        "ffn/out_weights/Adam:0", "ffn/out_weights/Adam_1:0",
        "ffn/out_weights:0"
    ]
    vars_list[2] = []
    grads = [[], [], []]
    grads[0] = [
        "clip_by_global_norm/clip_by_global_norm/_2:0",
        "clip_by_global_norm/clip_by_global_norm/_3:0"
    ]
    grads[1] = []
    grads[2] = [
        "clip_by_global_norm/clip_by_global_norm/_0:0",
        "clip_by_global_norm/clip_by_global_norm/_1:0"
    ]
    for i in range(3):
      taskgraph = g.taskgraphs[i]
      var = [ele.name for ele in taskgraph.get_variables(0)]
      grd = [ele.name for ele in taskgraph.gradients]
      list.sort(var)
      list.sort(grd)
      self.assertEqual(var, vars_list[i])
      self.assertEqual(grd, grads[i])

  def test_graph_with_local_clip_and_amp(self):
    config = epl.Config()
    config.amp.level = "O1"
    config.amp.loss_scale = 128
    config.cluster.colocate_split_and_replicate = True
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    g = Graph.get()
    loss = self._model_def()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads = [tf.clip_by_norm(grad, clip_norm=1.0) for grad in grads]
    optimizer.apply_gradients(list(zip(grads, tvars)))
    tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
        log_device_placement=False))

    # check taskgraph.
    self.assertTrue(len(g.taskgraphs) == 3)
    self.assertTrue(g.taskgraphs[0].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[1].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[2].local_num_replicas == 1)
    vars_list = [[], [], []]
    vars_list[0] = [
        "dense/bias/Adam:0", "dense/bias/Adam_1:0", "dense/bias:0",
        "dense/kernel/Adam:0", "dense/kernel/Adam_1:0", "dense/kernel:0"
    ]
    # TODO(jiangle.jl): Merge Taskgraph 1 and Taskgraph 2
    vars_list[1] = [
        "beta1_power:0", "beta2_power:0", "ffn/in_weights/Adam:0",
        "ffn/in_weights/Adam_1:0", "ffn/in_weights:0",
        "ffn/out_weights/Adam:0", "ffn/out_weights/Adam_1:0",
        "ffn/out_weights:0"
    ]
    vars_list[2] = []
    grads = [[], [], []]
    grads[0] = [
        "clip_by_norm_2:0",
        "clip_by_norm_3:0"
    ]
    grads[1] = []
    grads[2] = [
        "clip_by_norm:0",
        "clip_by_norm_1:0"
    ]
    for i in range(3):
      taskgraph = g.taskgraphs[i]
      var = [ele.name for ele in taskgraph.get_variables(0)]
      grd = [ele.name for ele in taskgraph.gradients]
      list.sort(var)
      list.sort(grd)
      self.assertEqual(var, vars_list[i])
      self.assertEqual(grd, grads[i])

  def test_graph_with_global_clip_and_amp(self):
    config = epl.Config()
    config.amp.level = "O1"
    config.amp.loss_scale = 128
    config.cluster.colocate_split_and_replicate = True
    epl.init(config)
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
    self.assertTrue(len(g.taskgraphs) == 3)
    self.assertTrue(g.taskgraphs[0].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[1].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[2].local_num_replicas == 1)
    vars_list = [[], [], []]
    vars_list[0] = [
        "dense/bias/Adam:0", "dense/bias/Adam_1:0", "dense/bias:0",
        "dense/kernel/Adam:0", "dense/kernel/Adam_1:0", "dense/kernel:0"
    ]
    # TODO(jiangle.jl): Merge Taskgraph 1 and Taskgraph 2
    vars_list[1] = [
        "beta1_power:0", "beta2_power:0", "ffn/in_weights/Adam:0",
        "ffn/in_weights/Adam_1:0", "ffn/in_weights:0",
        "ffn/out_weights/Adam:0", "ffn/out_weights/Adam_1:0",
        "ffn/out_weights:0"
    ]
    vars_list[2] = []
    grads = [[], [], []]
    grads[0] = [
        "clip_by_global_norm/clip_by_global_norm/_2:0",
        "clip_by_global_norm/clip_by_global_norm/_3:0"
    ]
    grads[1] = []
    grads[2] = [
        "clip_by_global_norm/clip_by_global_norm/_0:0",
        "clip_by_global_norm/clip_by_global_norm/_1:0"
    ]
    for i in range(3):
      taskgraph = g.taskgraphs[i]
      var = [ele.name for ele in taskgraph.get_variables(0)]
      grd = [ele.name for ele in taskgraph.gradients]
      list.sort(var)
      list.sort(grd)
      self.assertEqual(var, vars_list[i])
      self.assertEqual(grd, grads[i])

  def test_graph_with_local_clip_and_scale(self):
    conf = epl.Config()
    conf.cluster.colocate_split_and_replicate = True
    epl.init(conf)
    epl.set_default_strategy(epl.replicate(1))
    g = Graph.get()
    loss = self._model_def()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads = [tf.clip_by_norm(grad, clip_norm=1.0) for grad in grads]
    # Scale gradients manually
    grads = [grad * float(1 / 2) for grad in grads]
    optimizer.apply_gradients(list(zip(grads, tvars)))
    tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
        log_device_placement=False))

    # check taskgraph.
    self.assertTrue(len(g.taskgraphs) == 3)
    self.assertTrue(g.taskgraphs[0].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[1].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[2].local_num_replicas == 1)
    vars_list = [[], [], []]
    vars_list[0] = [
        "dense/bias/Adam:0", "dense/bias/Adam_1:0", "dense/bias:0",
        "dense/kernel/Adam:0", "dense/kernel/Adam_1:0", "dense/kernel:0"
    ]
    # TODO(jiangle.jl): Merge Taskgraph 1 and Taskgraph 2
    vars_list[1] = [
        "beta1_power:0", "beta2_power:0", "ffn/in_weights/Adam:0",
        "ffn/in_weights/Adam_1:0", "ffn/in_weights:0",
        "ffn/out_weights/Adam:0", "ffn/out_weights/Adam_1:0",
        "ffn/out_weights:0"
    ]
    vars_list[2] = []
    grads = [[], [], []]
    grads[0] = ["mul_2:0", "mul_3:0"]
    grads[1] = []
    grads[2] = ["mul:0", "mul_1:0"]
    for i in range(3):
      taskgraph = g.taskgraphs[i]
      var = [ele.name for ele in taskgraph.get_variables(0)]
      grd = [ele.name for ele in taskgraph.gradients]
      list.sort(var)
      list.sort(grd)
      self.assertEqual(var, vars_list[i])
      self.assertEqual(grd, grads[i])

  def test_graph_with_local_clip_and_scale_and_amp(self):
    config = epl.Config()
    config.amp.level = "O1"
    config.amp.loss_scale = 128
    config.cluster.colocate_split_and_replicate = True
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    g = Graph.get()
    loss = self._model_def()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads = [tf.clip_by_norm(grad, clip_norm=1.0) for grad in grads]
    # Scale gradients manually
    grads = [grad * float(1 / 2) for grad in grads]
    optimizer.apply_gradients(list(zip(grads, tvars)))
    tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
        log_device_placement=False))

    # check taskgraph.
    self.assertTrue(len(g.taskgraphs) == 3)
    self.assertTrue(g.taskgraphs[0].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[1].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[2].local_num_replicas == 1)
    vars_list = [[], [], []]
    vars_list[0] = [
        "dense/bias/Adam:0", "dense/bias/Adam_1:0", "dense/bias:0",
        "dense/kernel/Adam:0", "dense/kernel/Adam_1:0", "dense/kernel:0"
    ]
    # TODO(jiangle.jl): Merge Taskgraph 1 and Taskgraph 2
    vars_list[1] = [
        "beta1_power:0", "beta2_power:0", "ffn/in_weights/Adam:0",
        "ffn/in_weights/Adam_1:0", "ffn/in_weights:0",
        "ffn/out_weights/Adam:0", "ffn/out_weights/Adam_1:0",
        "ffn/out_weights:0"
    ]
    vars_list[2] = []
    grads = [[], [], []]
    grads[0] = ["mul_3:0", "mul_4:0"] if new_tf_version else ["mul_2:0", "mul_3:0"]
    grads[1] = []
    grads[2] = ["mul_1:0", "mul_2:0"] if new_tf_version else ["mul:0", "mul_1:0"]
    self.assertEqual(len(g.taskgraphs[0].gradients), 2)
    self.assertEqual(len(g.taskgraphs[2].gradients), 2)
    all_gradients = g.taskgraphs[0].gradients + g.taskgraphs[2].gradients
    self.assertEqual(sorted(all_gradients, key=lambda x: x.name), g.gradients)
    for i in range(3):
      taskgraph = g.taskgraphs[i]
      var = [ele.name for ele in taskgraph.get_variables(0)]
      grd = [ele.name for ele in taskgraph.gradients]
      list.sort(var)
      list.sort(grd)
      self.assertEqual(var, vars_list[i])

  def test_graph_with_global_clip_and_scale(self):
    config = epl.Config()
    config.cluster.colocate_split_and_replicate = True
    epl.init(config)
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
    self.assertTrue(len(g.taskgraphs) == 3)
    self.assertTrue(g.taskgraphs[0].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[1].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[2].local_num_replicas == 1)
    vars_list = [[], [], []]
    vars_list[0] = [
        "dense/bias/Adam:0", "dense/bias/Adam_1:0", "dense/bias:0",
        "dense/kernel/Adam:0", "dense/kernel/Adam_1:0", "dense/kernel:0"
    ]
    # TODO(jiangle.jl): Merge Taskgraph 1 and Taskgraph 2
    vars_list[1] = [
        "beta1_power:0", "beta2_power:0", "ffn/in_weights/Adam:0",
        "ffn/in_weights/Adam_1:0", "ffn/in_weights:0",
        "ffn/out_weights/Adam:0", "ffn/out_weights/Adam_1:0",
        "ffn/out_weights:0"
    ]
    vars_list[2] = []
    grads = [[], [], []]
    grads[0] = ["mul_2:0", "mul_3:0"]
    grads[1] = []
    grads[2] = ["mul:0", "mul_1:0"]
    for i in range(3):
      taskgraph = g.taskgraphs[i]
      var = [ele.name for ele in taskgraph.get_variables(0)]
      grd = [ele.name for ele in taskgraph.gradients]
      list.sort(var)
      list.sort(grd)
      self.assertEqual(var, vars_list[i])
      self.assertEqual(grd, grads[i])

  def test_graph_with_local_clip_after_allreduce(self):
    conf = Config()
    # Clip gradients after allreduce
    conf.communication.clip_after_allreduce = True
    conf.cluster.colocate_split_and_replicate = True
    epl.init(conf)
    epl.set_default_strategy(epl.replicate(1))
    g = Graph.get()
    loss = self._model_def()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)
    grads = [tf.clip_by_norm(grad, clip_norm=1.0) for grad in grads]
    optimizer.apply_gradients(list(zip(grads, tvars)))
    tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
        log_device_placement=False))

    # check taskgraph.
    self.assertTrue(len(g.taskgraphs) == 3)
    self.assertTrue(g.taskgraphs[0].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[1].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[2].local_num_replicas == 1)
    vars_list = [[], [], []]
    vars_list[0] = [
        "dense/bias/Adam:0", "dense/bias/Adam_1:0", "dense/bias:0",
        "dense/kernel/Adam:0", "dense/kernel/Adam_1:0", "dense/kernel:0"
    ]
    # TODO(jiangle.jl): Merge Taskgraph 1 and Taskgraph 2
    vars_list[1] = [
        "beta1_power:0", "beta2_power:0", "ffn/in_weights/Adam:0",
        "ffn/in_weights/Adam_1:0", "ffn/in_weights:0",
        "ffn/out_weights/Adam:0", "ffn/out_weights/Adam_1:0",
        "ffn/out_weights:0"
    ]
    vars_list[2] = []
    grads = [[], [], []]
    grads[0] = [
        "gradients/dense/BiasAdd_grad/BiasAddGrad:0",
        "gradients/dense/MatMul_grad/MatMul_1:0"
    ]
    grads[1] = []
    if new_tf_version:
      grads[2] = [
          "gradients/ffn/inter_outputs/MatMul_grad/Reshape_1:0",
          "gradients/ffn/outputs/MatMul_grad/Reshape_1:0"
      ]
    else:
      if six.PY2:
        grads[2] = [
            "gradients/ffn/inter_outputs/MatMul_grad/MatMul_1:0",
            "gradients/ffn/outputs/MatMul_grad/MatMul_1:0"
        ]
      else:
        grads[2] = [
            "gradients/ffn/inter_outputs/transpose_1_grad/transpose:0",
            "gradients/ffn/outputs/transpose_1_grad/transpose:0"
        ]

    for i in range(3):
      taskgraph = g.taskgraphs[i]
      var = [ele.name for ele in taskgraph.get_variables(0)]
      grd = [ele.name for ele in taskgraph.gradients]
      list.sort(var)
      list.sort(grd)
      self.assertEqual(var, vars_list[i])
      self.assertEqual(grd, grads[i])

  def test_graph_with_global_clip_after_allreduce(self):
    conf = Config()
    # Clip gradients after allreduce
    conf.communication.clip_after_allreduce = True
    conf.cluster.colocate_split_and_replicate = True
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
    self.assertTrue(len(g.taskgraphs) == 3)
    self.assertTrue(g.taskgraphs[0].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[1].local_num_replicas == 1)
    self.assertTrue(g.taskgraphs[2].local_num_replicas == 1)
    vars_list = [[], [], []]
    vars_list[0] = [
        "dense/bias/Adam:0", "dense/bias/Adam_1:0", "dense/bias:0",
        "dense/kernel/Adam:0", "dense/kernel/Adam_1:0", "dense/kernel:0"
    ]
    # TODO(jiangle.jl): Merge Taskgraph 1 and Taskgraph 2
    vars_list[1] = [
        "beta1_power:0", "beta2_power:0", "ffn/in_weights/Adam:0",
        "ffn/in_weights/Adam_1:0", "ffn/in_weights:0",
        "ffn/out_weights/Adam:0", "ffn/out_weights/Adam_1:0",
        "ffn/out_weights:0"
    ]
    vars_list[2] = []
    grads = [[], [], []]
    grads[0] = [
        "gradients/dense/BiasAdd_grad/BiasAddGrad:0",
        "gradients/dense/MatMul_grad/MatMul_1:0"
    ]
    grads[1] = []
    if new_tf_version:
      grads[2] = [
          "gradients/ffn/inter_outputs/MatMul_grad/Reshape_1:0",
          "gradients/ffn/outputs/MatMul_grad/Reshape_1:0"
      ]
    else:
      if six.PY2:
        grads[2] = [
            "gradients/ffn/inter_outputs/MatMul_grad/MatMul_1:0",
            "gradients/ffn/outputs/MatMul_grad/MatMul_1:0"
        ]
      else:
        grads[2] = [
            "gradients/ffn/inter_outputs/transpose_1_grad/transpose:0",
            "gradients/ffn/outputs/transpose_1_grad/transpose:0"
        ]

    for i in range(3):
      taskgraph = g.taskgraphs[i]
      var = [ele.name for ele in taskgraph.get_variables(0)]
      grd = [ele.name for ele in taskgraph.gradients]
      list.sort(var)
      list.sort(grd)
      self.assertEqual(var, vars_list[i])
      self.assertEqual(grd, grads[i])


# pylint: enable=missing-docstring,protected-access,unused-argument,arguments-differ
# pylint: enable=line-too-long,bad-continuation,unused-variable

if __name__ == "__main__":
  test.main()
