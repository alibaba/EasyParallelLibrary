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
"""Test for amp parallel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.framework.versions import __version__
from tensorflow.python.platform import test

import epl


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class AmpParallelTest(test.TestCase):

  def test_pipeline(self):
    epl.init(epl.Config({"amp.level": "O1", "amp.loss_scale": 128, "amp.debug_log": True}))
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                         iterator.initializer)
    x, _ = iterator.get_next()
    with epl.replicate(device_count=1):
      x = tf.layers.dense(inputs=x,
                          units=16,
                          activation=None,
                          name="stage1_dense1")
      x = tf.layers.dense(inputs=x, units=16, activation=None, name="stage1_dense2")
    with epl.replicate(device_count=1):
      dense1 = tf.layers.dense(inputs=x, units=16, activation=None, name="stage2_dense3")
      logits = tf.layers.dense(inputs=dense1, units=10, activation=None, name="stage2_dense4")
      loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = opt.minimize(loss, global_step=global_step)

    hooks = [tf.train.StopAtStepHook(last_step=5)]
    steps = []
    variables = []
    self.assertTrue(opt.get_slot_names() == ['m', 'v'])
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      unscale_ops = [op for op in epl.Graph.get().operations.values() if "UNSCALE_GRADIENTS" in op.name]
      stage1_ops = [op for op in unscale_ops if "stage1" in op.name]
      for op in stage1_ops:
        self.assertEqual(op.device, "/job:worker/replica:0/task:0/device:GPU:0")
        self.assertEqual(op.taskgraph.index, 0)
      stage2_ops = [op for op in unscale_ops if "stage2" in op.name]
      for op in stage2_ops:
        self.assertEqual(op.device, "/job:worker/replica:0/task:0/device:GPU:1")
        self.assertEqual(op.taskgraph.index, 1)
      while not sess.should_stop():
        train_opts = [loss, train_op, global_step]
        train_loss, _, step = sess.run(train_opts)


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  test.main()
