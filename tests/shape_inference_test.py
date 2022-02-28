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
"""Test for Shape Inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.utils.shape_inference import infer_shape, get_tf_tensor_shape
from epl.utils import constant

tf.logging.set_verbosity(tf.logging.INFO)


def parse_shape_from_metadata(run_metadata):
  """Parse shape from metadata."""
  dev_stats = run_metadata.step_stats.dev_stats
  name2shape = {}
  for dev_stat in dev_stats:
    node_stats = dev_stat.node_stats
    for ns in node_stats:
      if not ns.output: continue
      for i, output in enumerate(ns.output):
        shape = [int(s.size) for s in output.tensor_description.shape.dim]
        name2shape["{}:{}".format(ns.node_name, i)] = shape
  return name2shape


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class InferShapeTest(test.TestCase):
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
    tf.add_to_collection(constant.GC_COLLECTION_NAME, dense1)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    d0 = tf.shape(logits)[0]
    d1 = tf.shape(logits)[1]
    logits = tf.reshape(logits, [d0, d1])
    loss = tf.reduce_mean(logits)
    return loss

  def _get_tensor_shapes(self, tf_graph):
    infered_shapes = {}
    for op in tf_graph.get_operations():
      for i, t in enumerate(op.outputs):
        infered_shapes["{}:{}".format(op.name, i)] = get_tf_tensor_shape(t)
    return infered_shapes

  def test_simple_model(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    loss = self._model_def()
    original = self._get_tensor_shapes(tf.get_default_graph())
    unknow_list = [name for name in original if original[name] is None or \
        any(x is None for x in original[name])]
    infer_shape(tf.get_default_graph())
    infered_shapes = self._get_tensor_shapes(tf.get_default_graph())
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    loss = self._model_def()
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_opts = optimizer.minimize(loss, global_step=global_step)
    max_steps = 3
    hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      while not sess.should_stop():
        sess.run(train_opts, options=options, run_metadata=run_metadata)
    real_shapes = parse_shape_from_metadata(run_metadata)
    for name in unknow_list:
      if name in real_shapes and name in infered_shapes:
        self.assertEqual(infered_shapes[name], real_shapes[name])

  def test_attention(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))

    def model():
      num_x = np.random.randint(0, 10, (100, 272, 272)) \
          .astype(dtype=np.float32)
      num_x1 = np.random.randint(0, 10, (100, 272, 1024)) \
          .astype(dtype=np.float32)
      num_y = np.random.randint(0, 10, 100).astype(dtype=np.int32)
      dataset = tf.data.Dataset.from_tensor_slices((num_x, num_x1, num_y)) \
          .batch(4).repeat(20)
      iterator = dataset.make_initializable_iterator()
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           iterator.initializer)
      attention_mask, q_input, _ = iterator.get_next()

      batch_size = tf.shape(attention_mask)[0]
      q_seq_length = tf.shape(attention_mask)[1]
      kv_seq_length = tf.shape(attention_mask)[2]
      q_head_weight = tf.ones([1024, 16384])
      q_head_h = tf.einsum('bih,hx->bix', q_input, q_head_weight)
      num_attention_heads = 64
      attention_head_size = 256
      shape = [
          batch_size, q_seq_length, num_attention_heads, attention_head_size
      ]
      q_head_h_shape = tf.reshape(q_head_h, shape, name="test_reshape")
      return q_head_h_shape, q_head_h

    q1, q2 = model()
    original = self._get_tensor_shapes(tf.get_default_graph())
    unknow_list = [
        name for name in original
        if original[name] is None or any(x is None for x in original[name])
    ]
    infer_shape(tf.get_default_graph())
    infered_shapes = self._get_tensor_shapes(tf.get_default_graph())

    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    q1, q2 = model()
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    with tf.train.MonitoredTrainingSession() as sess:
      q1v, q2v = sess.run([q1, q2], options=options, run_metadata=run_metadata)
    real_shapes = parse_shape_from_metadata(run_metadata)
    for name in unknow_list:
      if name in real_shapes and name in infered_shapes:
        self.assertEqual(infered_shapes[name], real_shapes[name])


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  test.main()
