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
"""Test for Auto Gradient Checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.config import Config
from epl.runtime.gc.auto_gradient_checkpoint import get_entrance_exits_tensors, \
        filter_ops
from test_utils import fix_randomness
from test_utils import input_to_tensorarray

# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class AutoGCTest(test.TestCase):
  def _model_def(self):
    with epl.replicate(device_count=1):
      num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
      num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
      dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
          .batch(10).repeat(1)

      iterator = dataset.make_initializable_iterator()
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           iterator.initializer)
      x1, _ = iterator.get_next()
    with epl.replicate(device_count=1):
      x = tf.layers.dense(inputs=x1, units=16, activation=None, name="dense1")
      x2 = tf.layers.dense(inputs=x, units=16, activation=None, name="dense2")
    with epl.replicate(device_count=1):
      loss = tf.reduce_mean(x2)
    return x1, x2, loss

  def test_get_entrance_exits_tensors(self):
    epl.init()
    x1, x2, loss = self._model_def()
    ops = epl.Graph.get().taskgraphs[1].operations.forward_operations(0, 0)
    ops = [op.primitive_obj for op in ops]
    entrance_ts, exit_ts = get_entrance_exits_tensors(ops)
    self.assertEqual(len(entrance_ts), 1)
    self.assertEqual(len(exit_ts), 1)
    self.assertEqual(x1, entrance_ts[0])
    self.assertEqual(x2, exit_ts[0])

  def test_filter_ops(self):
    epl.init()
    x1, x2, loss = self._model_def()
    fwd_ops = tf.get_default_graph().get_operations()
    fwd_ops = filter_ops(fwd_ops)
    filtered_names = [
        "IteratorGetNext", "dense1/MatMul", "dense1/BiasAdd",
        "dense2/MatMul", "dense2/BiasAdd", "Mean"
    ]
    for op in fwd_ops:
      self.assertTrue(op.name in filtered_names)



  def _model_def_while_loop(self):
    fix_randomness()
    num_x = np.random.randint(0, 1, (500, 1, 1)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 1, 500).astype(dtype=np.int32)
    seq_len = 1
    hidden_dim = 1
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(1).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                         iterator.initializer)
    x, y = iterator.get_next()
    u = tf.get_variable(name='U', shape=[1, hidden_dim], dtype=tf.float32)
    b_u = tf.get_variable(name='b_U', shape=[hidden_dim], dtype=tf.float32)

    v = tf.get_variable(name='V', shape=[hidden_dim, 1], dtype=tf.float32)
    b_v = tf.get_variable(name='b_V', shape=[1], dtype=tf.float32)

    w = tf.get_variable(name='W', shape=[hidden_dim, hidden_dim], \
                        dtype=tf.float32)
    b_w = tf.get_variable(name='b_W', shape=[hidden_dim], dtype=tf.float32)
    input_ta = input_to_tensorarray(x, 1, seq_len)
    h = tf.TensorArray(tf.float32, seq_len+1, clear_after_read=False)
    h = h.write(0, tf.constant(np.zeros((1, hidden_dim)), dtype=tf.float32))
    output = tf.TensorArray(tf.float32, seq_len)
    time = tf.constant(0, dtype=tf.int32)

    def loop_body(time, hidden, output):
      input_step = input_ta.read(time)
      h_prev = hidden.read(time)
      hidden = hidden.write(time + 1, tf.tanh(tf.matmul(input_step, u) + \
                            b_u + tf.matmul(h_prev, w) + b_w))
      output = output.write(time, tf.matmul(hidden.read(time + 1), v) + b_v)
      return (time + 1, hidden, output)
    # build graph using while_loop
    loop_cond_fn = lambda time, _1, _2: time < seq_len
    final_state_ = tf.while_loop(cond=loop_cond_fn, body=loop_body, \
                                 loop_vars=(time, h, output), \
                                 back_prop=False)

    final_state = final_state_
    final_output = final_state[-1].read(seq_len-1)
    for i in range(10):
      final_output = tf.layers.dense(inputs=final_output, units=16)
    final_output = tf.layers.dense(inputs=final_output, units=1)
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=y, logits=tf.reshape(final_output, shape=[1]))
    return tf.reduce_mean(loss)

  def test_while_loop(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))

    loss = self._model_def_while_loop()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    opt = optimizer.minimize(loss)
    max_step = 5
    base = []
    with tf.train.MonitoredTrainingSession() as sess:
      for i in range(max_step):
        l, _ = sess.run([loss, opt])
        base.append(l)

    gc_loss = []
    config = Config()
    config.gradient_checkpoint.type = "auto"
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    loss = self._model_def_while_loop()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    opt = optimizer.minimize(loss)
    with tf.train.MonitoredTrainingSession() as sess:
      for i in range(max_step):
        l, _ = sess.run([loss, opt])
        gc_loss.append(l)
    operations = epl.Graph.get().operations
    sg_ops = sorted([name for name in operations if name.endswith("_gc_sg")])
    self.assertEqual(sg_ops, ['dense_1/BiasAdd_gc_sg', \
                                     'dense_3/BiasAdd_gc_sg', \
                                     'dense_5/BiasAdd_gc_sg', \
                                     'dense_7/BiasAdd_gc_sg'])
    for r1, r2 in list(zip(base, gc_loss)):
      self.assertTrue(abs(r1 - r2) < 1e-5,
                      "{} and {} not equal".format(base, gc_loss))


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  test.main()
