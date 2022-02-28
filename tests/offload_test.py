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
"""Test for offload weight."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from test_utils import fix_randomness


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class OffloadTest(test.TestCase):
  def _model_def(self):
    fix_randomness()
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None, name="densetest")
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return [loss, train_op, global_step]

  def offload_case(self, config, ckpt_dir=None):
    fix_randomness()
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    res = []
    max_steps = 5
    with tf.Graph().as_default():
      train_opts = self._model_def()
      with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir) as sess:
        graph = tf.get_default_graph()
        name2dev = {o.name: o.device for o in graph.get_operations() if
                    o.type == 'VariableV2'}
        for i in range(max_steps):
          train_loss, _, step = sess.run(train_opts)
          res.append((train_loss, step))
    return res, name2dev

  def test_offload_v0(self):
    config = epl.Config()
    res0, name2dev = self.offload_case(config)
    for name, dev in name2dev.items():
      self.assertEqual(dev, '/job:worker/replica:0/task:0/device:GPU:0')
    config = epl.Config()
    config.offload.level = "v0"
    res1, name2dev = self.offload_case(config)
    for name, dev in name2dev.items():
      self.assertEqual(dev, '/job:worker/replica:0/task:0/device:CPU:0')
    for r1, r2 in list(zip(res0, res1)):
      r1 = r1[0]
      r2 = r2[0]
      self.assertTrue(abs(r1 - r2) < 1e-6, '{} vs {} fail'.format(r1, r2))

  def test_save_load_ckpt(self):
    d = tempfile.mkdtemp()
    config = epl.Config()
    config.offload.level = "v0"
    res1, name2dev = self.offload_case(config, d)
    step1 = [r[1] for r in res1]
    self.assertEqual(step1, [1, 2, 3, 4, 5])
    config = epl.Config()
    res0, name2dev = self.offload_case(config, d)
    shutil.rmtree(d)
    step0 = [r[1] for r in res0]
    self.assertEqual(step0, [5, 6, 7, 8, 9])


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  test.main()
