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
"""Test for flops statistics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base
from tensorflow.python.platform import test

import epl
from epl.profiler.flops import FlopsProfilerHook
from epl.profiler.flops import profile_flops

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
    """Definition for weights."""
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

  def call(self, inputs_1, inputs_2):  # pylint: disable=arguments-differ
    """Call einsum to implement ffn."""
    with epl.split():
      intermediate = tf.einsum('EGCM,EMH->EGCH',
                               inputs_1,
                               self.in_weights,
                               name="inter_outputs")
      # activation function
      activated_inters = self.activation_fn(intermediate)

      # output forward
      outputs = tf.einsum('EGCH,EHM->EGCM',
                          activated_inters,
                          self.out_weights,
                          name="outputs")
      outputs = tf.einsum('GSEC,EGCM->GSM',
                          inputs_2,
                          outputs,
                          name="combined_outputs")
      outputs = tf.reshape(outputs, [-1, 1280])
      return outputs


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class SplitTest(test.TestCase):

  def _split_with_apply(self, cmds, use_hook):
    config = epl.Config()
    config.cluster.colocate_split_and_replicate = True
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    num_x1 = np.random.randint(0, 10, (500, 10, 20, 16)) \
        .astype(dtype=np.float32)
    num_x2 = np.random.randint(0, 10, (500, 8, 10, 20)) \
        .astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x1, num_x2, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x1, x2, _ = iterator.get_next()
    x1.set_shape([10, 10, 20, 16])
    x2.set_shape([10, 8, 10, 20])
    ffn = FFN()
    dense1 = ffn(x1, x2)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.00001, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    hooks = []
    if use_hook:
      hooks = [FlopsProfilerHook(cmds=cmds)]
    run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      if not use_hook:
        run_metadata = tf.RunMetadata()
        train_loss, _, step = sess.run([loss, train_op, global_step],
                                       options=run_option,
                                       run_metadata=run_metadata)
        flops = profile_flops(sess, run_metadata, cmds)
        self.assertEqual(flops, 18228514)
      else:
        train_loss, _, step = sess.run([loss, train_op, global_step])

  def test_split(self):
    for cmds in [["scope"], ["op", "scope", "graph"]]:
      for use_hook in [True, False]:
        self._split_with_apply(cmds=cmds, use_hook=use_hook)

  def test_dense(self):
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None, name="dense1")
    cmds = ['scope']
    hooks = [FlopsProfilerHook(cmds=cmds)]
    run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      run_metadata = tf.RunMetadata()
      sess.run([x], options=run_option,
               run_metadata=run_metadata)
      flops = profile_flops(sess, run_metadata, ['op'])
      self.assertEqual(flops, 3681)


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  test.main()
