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
"""Test for amp."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion as Version
import numpy as np

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_50
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework.versions import __version__
from tensorflow.python.platform import test

import epl
from epl.utils import constant
from test_utils import fix_randomness


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class AmpTest(test.TestCase):
  def _model_def(self, tfamp=False, loss_scale=128, auto_gc=False):
    fix_randomness()
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    if not auto_gc:
      x = tf.layers.dense(inputs=x, units=16, activation=None)
      x = tf.layers.dense(inputs=x, units=16, activation=None)
      tf.add_to_collection("checkpoints", x)
    else:
      for i in range(5):
        with tf.name_scope('s1/s2/s3/s4_layer{}'.format(i)):
          for j in range(5):
            x = tf.layers.dense(inputs=x, units=16, use_bias=False, activation='relu')
            x = tf.clip_by_norm(x, 2.0)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    if tfamp:
      from tensorflow.python.training.experimental import mixed_precision
      enable_mixed_precision_graph_rewrite = (
          mixed_precision.enable_mixed_precision_graph_rewrite_v1)
      optimizer = enable_mixed_precision_graph_rewrite(optimizer,
                                                       loss_scale=loss_scale)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return [loss, train_op, global_step]

  def create_session(self):
    max_steps = 3
    hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
    config_proto = tf.ConfigProto()
    config_proto.graph_options.rewrite_options.arithmetic_optimization = \
      rewriter_config_pb2.RewriterConfig.OFF
    sess = tf.train.MonitoredTrainingSession(hooks=hooks, config=config_proto)
    return sess

  def test_amp_fix(self):
    tfamp = Version(__version__) >= Version("1.15.0")
    if tfamp:
      epl.init()
      epl.set_default_strategy(epl.replicate(1))
      losses1 = []
      with tf.Graph().as_default():
        train_opts = self._model_def(tfamp, loss_scale=128)
        with self.create_session() as sess:
          while not sess.should_stop():
            train_loss, _, step = sess.run(train_opts)
            losses1.append(train_loss)
    config = epl.Config({"amp.level": "O1", "amp.loss_scale": 128})
    tfamp = False
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    losses2 = []
    with tf.Graph().as_default():
      train_opts = self._model_def(tfamp, loss_scale=128)
      collect = True
      with self.create_session() as sess:
        while not sess.should_stop():
          if collect:
            train_loss = self._check_fp16_node(sess, train_opts)
            collect = False
          else:
            train_loss, _, step = sess.run(train_opts)
          losses2.append(train_loss)

    if tfamp:
      for l1, l2 in list(zip(losses1, losses2)):
        self.assertTrue(abs(l1 - l2) < 1e-6)

  def test_auto_dynamic(self):
    tfamp = Version(__version__) >= Version("1.15.0")
    if tfamp:
      epl.init()
      epl.set_default_strategy(epl.replicate(1))
      losses1 = []
      with tf.Graph().as_default():
        train_opts = self._model_def(tfamp, loss_scale="dynamic")
        with self.create_session() as sess:
          while not sess.should_stop():
            train_loss, _, step = sess.run(train_opts)
            losses1.append(train_loss)
    tfamp = False
    config = epl.Config({"amp.level": "O1", "amp.loss_scale": "dynamic"})
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    losses2 = []
    with tf.Graph().as_default():
      train_opts = self._model_def(tfamp, loss_scale="dynamic")
      with self.create_session() as sess:
        collect = True
        while not sess.should_stop():
          if collect:
            train_loss = self._check_fp16_node(sess, train_opts)
            collect = False
          else:
            train_loss, _, step = sess.run(train_opts)
          losses2.append(train_loss)
    if tfamp:
      for l1, l2 in list(zip(losses1, losses2)):
        self.assertTrue(abs(l1 - l2) < 1e-6)

  def _check_fp16_node(self, sess, train_opts):
    run_metadata = tf.RunMetadata()
    run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    train_loss, _, step = sess.run(train_opts, options=run_option, run_metadata=run_metadata)
    all_streams = [stats for stats in run_metadata.step_stats.dev_stats if stats.device == '/device:GPU:0/stream:all'][0]
    scale_nodes = [node.node_name for node in all_streams.node_stats if constant.EPL_AMP_SUFFIX in node.node_name]
    self.assertTrue(any("dense/MatMul_EPL_AMP_float16:MatMul" in node for node in scale_nodes))
    return train_loss

  def test_auto_dynamic_gc(self):
    tfamp = Version(__version__) >= Version("1.15.0")
    if tfamp:
      epl.init()
      epl.set_default_strategy(epl.replicate(1))
      losses1 = []
      with tf.Graph().as_default():
        train_opts = self._model_def(tfamp, loss_scale="dynamic")
        with self.create_session() as sess:
          while not sess.should_stop():
            train_loss, _, step = sess.run(train_opts)
            losses1.append(train_loss)
    config = epl.Config({"amp.level": "O1",
                         "amp.loss_scale": "dynamic",
                         "amp.debug_log": True,
                         "gradient_checkpoint.type": "collection"})
    tfamp = False
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    losses2 = []
    collect = True
    with tf.Graph().as_default():
      train_opts = self._model_def(tfamp, loss_scale="dynamic")
      with self.create_session() as sess:
        while not sess.should_stop():
          if collect:
            train_loss = self._check_fp16_node(sess, train_opts)
            collect = False
          else:
            train_loss, _, step = sess.run(train_opts)

          losses2.append(train_loss)
    if tfamp:
      for l1, l2 in list(zip(losses1, losses2)):
        self.assertTrue(abs(l1 - l2) < 1e-6)

  def test_auto_dynamic_gc_auto(self):
    tfamp = Version(__version__) >= Version("1.15.0")
    if tfamp:
      epl.init()
      epl.set_default_strategy(epl.replicate(1))
      losses1 = []
      with tf.Graph().as_default():
        train_opts = self._model_def(tfamp, loss_scale="dynamic", auto_gc=True)
        with self.create_session() as sess:
          while not sess.should_stop():
            train_loss, _, step = sess.run(train_opts)
            losses1.append(train_loss)
    config = epl.Config({"amp.level": "O1",
                         "amp.loss_scale": "dynamic",
                         "amp.debug_log": True,
                         "gradient_checkpoint.type": "auto"})
    tfamp = False
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    losses2 = []
    collect = True
    with tf.Graph().as_default():
      train_opts = self._model_def(tfamp, loss_scale="dynamic", auto_gc=True)
      with self.create_session() as sess:
        gc_tensors = sorted([t.name for t in epl.Graph.get().gc_tensors])
        self.assertEqual(len(gc_tensors), 6)
        expected_names = ['s1/s2/s3/s4_layer{}/clip_by_norm_4:0'.format(i) for i in range(5)]
        expected_names = ['IteratorGetNext:0'] + expected_names
        self.assertEqual(gc_tensors, expected_names)
        while not sess.should_stop():
          if collect:
            train_loss = self._check_fp16_node(sess, train_opts)
            collect = False
          else:
            train_loss, _, step = sess.run(train_opts)
          losses2.append(train_loss)
    if tfamp:
      for l1, l2 in list(zip(losses1, losses2)):
        self.assertTrue(abs(l1 - l2) < 1e-6)

  def test_resnet_gc_amp(self):
    def _resnet_fn():
      dataset = tf.data.Dataset.from_tensor_slices(
          (tf.ones(shape=[100 * 64, 224, 224, 3],
                   dtype=tf.float32), tf.ones(shape=[
                       100 * 64,
                   ], dtype=tf.int64)))
      dataset = dataset.repeat()
      dataset = dataset.batch(2).prefetch(buffer_size=2)
      (images, labels) = dataset.make_one_shot_iterator().get_next()
      logits = resnet_v1_50(images, num_classes=10)[1]['predictions']
      logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                    logits=logits,
                                                    weights=1.0)
      return loss

    config = epl.Config({"gradient_checkpoint.type": "auto",
                         "amp.level": "o1",
                         "amp.loss_scale": 1000,
                         "amp.debug_log": True})
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    loss = _resnet_fn()
    opt = tf.train.AdamOptimizer(0.0001, name='adam')
    global_step = tf.train.get_or_create_global_step()
    train_opts = opt.minimize(loss, global_step=global_step, name='train')
    with tf.train.MonitoredTrainingSession() as sess:
      sess.run(train_opts)

# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  test.main()
