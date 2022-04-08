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
"""Test for multiple optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from test_utils import fix_randomness


def get_nested_optimizer():
  optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
  optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 0.5)
  optimizer = tf.contrib.opt.MovingAverageOptimizer(
      optimizer,
      average_decay=0.1,
      num_updates=2)
  return optimizer

def _train_fn(bs, lr_decay=False, max_steps=5, optimizer=None, enable_epl=True):
  """Define train function."""
  fix_randomness()
  num_x = np.random.randint(0, 10, (500, 20)).astype(dtype=np.float32)
  num_y = np.random.randint(0, 10, 500).astype(dtype=np.int64)
  dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
                           .batch(bs).repeat(1)
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  x, labels = iterator.get_next()
  for _ in range(10):
    x = tf.layers.dense(x, 10, use_bias=False)
  logits = x
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  if enable_epl:
    epl.add_to_collection(loss, epl.GraphKeys.GLOBAL_MEAN_OBJECTS)
  tf.summary.scalar('loss', loss)
  global_step = tf.train.get_or_create_global_step()
  learning_rate = 0.001
  if lr_decay:
    learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                               max_steps, 0.96,
                                               staircase=False)
  if optimizer is None:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  else:
    optimizer = optimizer
  train_op = optimizer.minimize(loss, global_step=global_step)
  merged = tf.summary.merge_all()
  hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
  losses = []
  with tf.train.MonitoredTrainingSession(
      hooks=hooks) as sess:
    while not sess.should_stop():
      fetches = [loss, train_op, global_step, merged]
      res = sess.run(fetches)
      train_loss = res[0]
      losses.append(train_loss)
      print("step: {} loss: {}".format(res[2], train_loss))
  return losses, optimizer

# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class MultiOptimizerTest(test.TestCase):
  @classmethod
  def setUpClass(cls): # pylint: disable=invalid-name
    with tf.Graph().as_default():
      optimizer = get_nested_optimizer()
      cls.base, _ = _train_fn(bs=2, lr_decay=False, max_steps=5, optimizer=optimizer, enable_epl=False)


  def test_nested_optimizer(self):
    epl.init(epl.Config({"communication.clip_after_allreduce": True}))
    epl.set_default_strategy(epl.replicate(1))
    with tf.Graph().as_default():
      optimizer = get_nested_optimizer()
      res2, _ = _train_fn(bs=1, lr_decay=False, max_steps=5, optimizer=optimizer)
      self.assertEqual(len(epl.Graph.get().gradients), 10)

  def test_nested_optimizer_clip_after_allreduce(self):
    epl.init(epl.Config({"communication.clip_after_allreduce": False}))
    epl.set_default_strategy(epl.replicate(1))
    with tf.Graph().as_default():
      optimizer = get_nested_optimizer()
      res2, _ = _train_fn(bs=1, lr_decay=False, max_steps=5, optimizer=optimizer)
      self.assertEqual(len(epl.Graph.get().gradients), 10)

# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  test.main()
