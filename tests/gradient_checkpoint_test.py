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
"""Test for Env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion as Version
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.python.platform import test
from tensorflow.python.framework.versions import __version__
from tensorflow.python.ops.rnn import raw_rnn
import tensorflow.contrib.rnn as rnn

import epl
from epl.config import Config
from epl.utils import constant
from epl.runtime.gc.gradient_checkpoint import capture_ops
from test_utils import fix_randomness

tf.logging.set_verbosity(tf.logging.INFO)

def lstm2_model(features):
  """lstm model define."""
  n_outputs = 1
  seq_len = 161
  n_inputs = seq_len - n_outputs
  x = tf.reshape(features, [-1, n_inputs, 1])
  lstm_cell1 = rnn.BasicLSTMCell(n_inputs*2, forget_bias=1.0)
  lstm_cell2 = rnn.BasicLSTMCell(n_inputs//2, forget_bias=1.0)
  lstm_cells = rnn.MultiRNNCell([lstm_cell1, lstm_cell2])
  outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_cell1, lstm_cell2,
                                               x, dtype=tf.float32)
  outputs = outputs[0]
  outputs = outputs[:, (n_inputs-1):, :] # last one only

  # 3. flatten lstm output and pass through a dense layer
  lstm_flat = tf.reshape(outputs, [-1, lstm_cells.output_size])
  h1 = tf.layers.dense(lstm_flat, lstm_cells.output_size//2,
                       activation=tf.nn.relu)
  predictions = tf.layers.dense(h1, 1, activation=None) # (?, 1)
  return predictions


def raw_rnn_fn(features):
  """raw rnn model."""
  input_depth = 1
  max_time = 2
  batch_size = 80
  #N_OUTPUTS
  inputs = tf.reshape(features, [max_time, batch_size, 1])
  num_units = 80
  inputs_ta = tf.TensorArray(dtype=tf.float32, size=max_time)
  inputs_ta = inputs_ta.unstack(inputs)
  cell = tf.contrib.rnn.LSTMCell(num_units)
  def loop_fn(time, cell_output, cell_state, loop_state): # pylint: disable=missing-docstring,unused-argument
    emit_output = cell_output
    if cell_output is None:
      next_cell_state = cell.zero_state(batch_size, tf.float32)
    else:
      next_cell_state = cell_state
    elements_finished = (time >= max_time)
    finished = tf.reduce_all(elements_finished)
    next_input = tf.cond(
        finished,
        lambda: tf.zeros([batch_size, input_depth], dtype=tf.float32),
        lambda: inputs_ta.read(time))
    next_loop_state = None
    return (elements_finished, next_input, next_cell_state,
            emit_output, next_loop_state)
  outputs_ta, _, _ = raw_rnn(cell, loop_fn)
  outputs = outputs_ta.stack()
  return outputs

def _model_def(config=None, use_raw_rnn=False, use_lstm=False):
  """define model."""
  if config is None:
    config = epl.Config()
  config.gradient_checkpoint.check_gradients = True
  epl.init(config)
  epl.set_default_strategy(epl.replicate(1))
  dropout_keep_prob = 0.5
  fix_randomness()
  num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
  num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
  dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
      .batch(10).repeat(1)
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  x, _ = iterator.get_next()
  x = tf.layers.dense(inputs=x, units=16, activation=None, name="dense0")
  x = tf.layers.dense(inputs=x, units=16, activation=None, name="dense1")
  x = layers.Dropout(dropout_keep_prob, name="dropout0")(x, True)
  dense1 = tf.layers.dense(inputs=x,
                           units=16,
                           activation=None,
                           name="dense2")
  tf.add_to_collection(constant.GC_COLLECTION_NAME, dense1)
  if use_raw_rnn:
    dense1 = raw_rnn_fn(dense1)
  if use_lstm:
    dense1 = lstm2_model(dense1)
  dense1 = layers.Dropout(dropout_keep_prob, name="dropout1")(dense1, True)
  logits = tf.layers.dense(inputs=dense1,
                           units=10,
                           activation=None,
                           name="dense3")
  loss = tf.reduce_mean(logits)
  global_step = tf.train.get_or_create_global_step()
  optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
  train_op = optimizer.minimize(loss, global_step=global_step)
  max_steps = 5
  hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
  train_opts = [loss, train_op, global_step]
  res = []
  with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
    while not sess.should_stop():
      train_loss, _, _ = sess.run(train_opts)
      res.append(train_loss)
  return train_opts, hooks, res


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class GradientCheckpointTest(test.TestCase):

  @classmethod
  def setUpClass(cls): # pylint: disable=invalid-name
    train_opts, hooks, res = _model_def()
    cls.baseline = res

  def test_gc_collection(self):
    config = Config({"gradient_checkpoint.type": constant.GC_COLLECTION})
    config.gradient_checkpoint.check_gradients = True
    _, _, gc_loss = _model_def(config)
    for r1, r2 in list(zip(gc_loss, self.baseline)):
      self.assertTrue(abs(r1 - r2) < 1e-5,
                      "{} and {} not equal".format(gc_loss, self.baseline))

  def test_gc_collection_pipeline(self):
    config = Config({"gradient_checkpoint.type": constant.GC_COLLECTION})
    config.gradient_checkpoint.check_gradients = True
    config.pipeline.num_micro_batch = 1
    _, _, gc_loss = _model_def(config)
    for r1, r2 in list(zip(gc_loss, self.baseline)):
      self.assertTrue(abs(r1 - r2) < 1e-5,
                      "{} and {} not equal".format(gc_loss, self.baseline))

  def test_gc_ignore_last_block(self):
    config = Config({"gradient_checkpoint.type": constant.GC_COLLECTION})
    config.gradient_checkpoint.check_gradients = True
    train_opts, hooks, _ = _model_def(config)
    checkpoints = tf.get_collection("checkpoints")
    operations = epl.Graph.get().operations
    sg_ops = [name for name in operations if name.endswith("_gc_sg")]
    recompute = [name for name in operations if
                 name.startswith("EPL_GRADIENT_CHECKPOINTS")]
    self.assertEqual(sg_ops, ['dense2/BiasAdd_gc_sg'])
    for name in recompute:
      self.assertTrue('dropout1' not in name)
      self.assertTrue('RandomUniform' not in name)
      self.assertTrue("dense3" not in name)

  def test_auto(self):
    config = Config({"gradient_checkpoint.type": constant.GC_AUTO})
    config.gradient_checkpoint.check_gradients = True
    train_opts, hooks, _ = _model_def(config)
    operations = epl.Graph.get().operations
    sg_ops = [name for name in operations if name.endswith("_gc_sg")]
    recompute = [name for name in operations if
                 name.startswith("EPL_GRADIENT_CHECKPOINTS")]
    self.assertTrue("dense0/BiasAdd_gc_sg" in sg_ops, sg_ops)
    new_tf_version = \
        Version(__version__) < Version("2.0") and \
        Version(__version__) >= Version("1.14.0")
    name = "dropout0/dropout/mul_1_gc_sg" if new_tf_version else \
        "dropout0/dropout/mul_gc_sg"
    self.assertTrue(name in sg_ops)

  def test_raw_rnn(self):
    config = Config({"gradient_checkpoint.type": constant.GC_COLLECTION})
    config.gradient_checkpoint.check_gradients = True
    train_opts, hooks, _ = _model_def(config, use_raw_rnn=True)
    checkpoints = tf.get_collection("checkpoints")
    operations = epl.Graph.get().operations
    sg_ops = [name for name in operations if name.endswith("_gc_sg")]
    recompute = [name for name in operations if
                 name.startswith("EPL_GRADIENT_CHECKPOINTS")]
    self.assertEqual(sg_ops, ['dense2/BiasAdd_gc_sg'])
    for name in recompute:
      self.assertTrue('dropout1' not in name)
      self.assertTrue('RandomUniform' not in name)
      self.assertTrue("dense3" not in name)

  def test_lstm(self):
    config = Config({"gradient_checkpoint.type": constant.GC_COLLECTION})
    config.gradient_checkpoint.check_gradients = True
    train_opts, hooks, _ = _model_def(config, use_lstm=True)
    checkpoints = tf.get_collection("checkpoints")
    operations = epl.Graph.get().operations
    sg_ops = [name for name in operations if name.endswith("_gc_sg")]
    recompute = [name for name in operations if
                 name.startswith("EPL_GRADIENT_CHECKPOINTS")]
    self.assertEqual(sg_ops, ['dense2/BiasAdd_gc_sg'])
    for name in recompute:
      self.assertTrue('dropout1' not in name)
      self.assertTrue('RandomUniform' not in name)
      self.assertTrue("dense3" not in name)

  def test_capture_ops(self):
    with capture_ops("TEMP") as temp_ops: # pylint: disable=too-many-function-args
      tf.ones([1, 2], name='a')
      tf.zeros([1, 2], name='b')
    temp_names = [o.name for o in temp_ops]
    self.assertTrue('TEMP/a' in temp_names)
    self.assertTrue('TEMP/b' in temp_names)

  def test_stop_gradients_collection(self):
    config = epl.Config()
    config.gradient_checkpoint.type = "collection"
    config.gradient_checkpoint.check_gradients = True
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, name='d1')
    tf.add_to_collection(constant.GC_COLLECTION_NAME, x)
    x = tf.stop_gradient(x)
    x = tf.layers.dense(inputs=x, units=16, name='d2')
    x = tf.layers.dense(inputs=x, units=16, name='d3')
    tf.add_to_collection(constant.GC_COLLECTION_NAME, x)
    x = tf.layers.dense(inputs=x, units=16)
    x = tf.layers.dense(inputs=x, units=16, name='d4')
    tf.add_to_collection(constant.GC_COLLECTION_NAME, x)
    dense1 = tf.layers.dense(inputs=x, units=16)
    logits = tf.layers.dense(inputs=dense1, units=10)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    with tf.train.MonitoredTrainingSession() as sess:
      sess.run(train_op)
    operations = epl.Graph.get().operations
    sg_ops = [name for name in operations if name.endswith("_gc_sg")]
    sg_ops.sort()
    self.assertEqual(sg_ops, ['d3/BiasAdd_gc_sg', 'd4/BiasAdd_gc_sg'])


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  test.main()
