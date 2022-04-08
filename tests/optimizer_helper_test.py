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
"""Test for gradient accumulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from test_utils import fix_randomness

# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class OptimizerHelperTest(test.TestCase):

  @classmethod
  def setUpClass(cls): # pylint: disable=invalid-name
    cls.port = 9010

  def model_def(self):
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                         iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None, name="dense1")
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = opt.minimize(loss, global_step=global_step)
    return opt, train_op, loss, global_step

  def gradient_accumulation_test(self, num_apply_group,
                                 opt_clip_after_allreduce):
    self.port += 1
    config = epl.Config()
    config.optimizer.num_apply_group = num_apply_group
    config.communication.clip_after_allreduce = opt_clip_after_allreduce
    config.pipeline.num_micro_batch = 5
    epl.init(config)
    opt, train_op, loss, global_step = self.model_def()

    hooks = [tf.train.StopAtStepHook(last_step=20)]
    steps = []
    variables = []
    self.assertTrue(opt.get_slot_names() == ['accum', 'm', 'v'],
                    opt.get_slot_names())
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      while not sess.should_stop():
        weights = sess.graph.get_tensor_by_name('dense1/kernel:0')
        train_opts = [loss, train_op, global_step, weights]
        train_loss, _, step, vs = sess.run(train_opts)
        steps.append(step)
        variables.append(vs)
    for i in range(4):
      mb_var = variables[i * 5]
      for mb in range(5):
        local_step = i * 5 + mb
        self.assertEqual(steps[local_step], i)
      for mb in range(4):
        local_step = i * 5 + mb
        self.assertTrue((variables[local_step] == mb_var).all())
    self.assertEqual(len(epl.Graph.get().gradients), 8)
    self.assertTrue(any("epl_apply_grad_ga" not in op_name for op_name \
                    in epl.Graph.get().operations))

  def test_gradient_accumulation_pipeline(self):
    for num_apply_group in [1, 3]:
      for opt_clip_after_allreduce in [True, False]:
        self.gradient_accumulation_test(num_apply_group,
                                        opt_clip_after_allreduce)

  def pipeline_test(self, num_apply_group=1):
    config = epl.Config()
    config.optimizer.num_apply_group = num_apply_group
    config.communication.clip_after_allreduce = False
    config.pipeline.num_micro_batch = 3
    epl.init(config)
    self.port += 1
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
                          name="dense1")
      x = tf.layers.dense(inputs=x, units=16, activation=None)
    with epl.replicate(device_count=1):
      dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
      logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
      loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = opt.minimize(loss, global_step=global_step)

    hooks = [tf.train.StopAtStepHook(last_step=5)]
    steps = []
    variables = []
    self.assertTrue(opt.get_slot_names() == ['m', 'v'])
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      while not sess.should_stop():
        weights = sess.graph.get_tensor_by_name('dense1/kernel:0')
        train_opts = [loss, train_op, global_step, weights]
        train_loss, _, step, vs = sess.run(train_opts)
        steps.append(step)
        variables.append(vs)
    self.assertEqual(steps, [0, 1, 2, 3, 4])
    if num_apply_group == 1:
      self.assertTrue(all("epl_apply_grad" not in op_name for op_name \
                      in epl.Graph.get().operations))
    self.assertTrue(any("EPL_MICRO_BATCH_2" in op_name for op_name \
                    in epl.Graph.get().operations))

  def test_pipeline(self):
    for num_apply_group in [1, 3]:
      self.pipeline_test(num_apply_group)

  def test_group_only(self):
    config = epl.Config()
    config.optimizer.num_apply_group = 3
    config.communication.clip_after_allreduce = False
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    with tf.Graph().as_default():
      opt, train_op, loss, global_step = self.model_def()
      hooks = [tf.train.StopAtStepHook(last_step=5)]
      steps = []
      variables = []
      self.assertTrue(opt.get_slot_names() == ['m', 'v'])
      with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        while not sess.should_stop():
          weights = sess.graph.get_tensor_by_name('dense1/kernel:0')
          train_opts = [loss, train_op, global_step, weights]
          train_loss, _, step, vs = sess.run(train_opts)
          steps.append(step)
          variables.append(vs)
      g = epl.Graph.get()
    self.assertEqual(steps, [0, 1, 2, 3, 4])
    self.assertTrue(any("epl_apply_grad_group" in op_name for op_name \
                    in g.operations))

  def test_ga_opt_clip_after_allreduce(self):
    config = epl.Config()
    config.optimizer.num_apply_group = 2
    config.communication.clip_after_allreduce = True
    config.pipeline.num_micro_batch = 5
    epl.init(config)
    self.port += 1
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                         iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None, name="dense1")
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    opt = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = opt.minimize(loss, global_step=global_step)
    hooks = [tf.train.StopAtStepHook(last_step=2)]
    steps = []
    variables = []
    self.assertTrue(opt.get_slot_names() == ['accum', 'm', 'v'])
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      while not sess.should_stop():
        weights = sess.graph.get_tensor_by_name('dense1/kernel:0')
        train_opts = [loss, train_op, global_step, weights]
        train_loss, _, step, vs = sess.run(train_opts)
        steps.append(step)
        variables.append(vs)
    self.assertEqual(steps, [0] * 5 + [1] * 5)
    self.assertTrue(any("epl_apply_grad_ga_group" in op_name for op_name \
                    in epl.Graph.get().operations))

  def _train_fn(self, bs, lr_decay=False, max_steps=5):
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
    epl.add_to_collection(loss, epl.GraphKeys.GLOBAL_MEAN_OBJECTS)
    tf.summary.scalar('loss', loss)
    global_step = tf.train.get_or_create_global_step()
    learning_rate = 0.001
    if lr_decay:
      learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                                 max_steps, 0.96,
                                                 staircase=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
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

  def test_group_apply_loss(self):
    config = epl.Config()
    config.optimizer.num_apply_group = 5
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    with tf.Graph().as_default():
      result1, _ = self._train_fn(bs=4)
    config = epl.Config()
    config.optimizer.num_apply_group = 1
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    with tf.Graph().as_default():
      result2, _ = self._train_fn(bs=4)
    for r1, r2 in list(zip(result1, result2)):
      self.assertTrue(abs(r1-r2) <= 1e-6, "{} and {} not equal" \
                      .format(result1, result2))

  def test_group_apply_loss_decay(self):
    config = epl.Config()
    config.optimizer.num_apply_group = 5
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    with tf.Graph().as_default():
      result1, _ = self._train_fn(bs=4, lr_decay=True)

    config = epl.Config()
    config.optimizer.num_apply_group = 1
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    with tf.Graph().as_default():
      result2, _ = self._train_fn(bs=4, lr_decay=True)
    for r1, r2 in list(zip(result1, result2)):
      self.assertTrue(abs(r1-r2) <= 1e-6, "{} and {} not equal" \
                      .format(result1, result2))

  def compare_gradient_accumulation(self, mb=1, apply_num=1):
    config = epl.Config()
    config.pipeline.num_micro_batch = 1
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    max_step = 10
    with tf.Graph().as_default():
      result1, opt = self._train_fn(bs=2*mb, lr_decay=True, max_steps=max_step)

    config = epl.Config()
    config.optimizer.num_apply_group = apply_num

    config.pipeline.num_micro_batch = mb
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    result2, opt = self._train_fn(bs=2, lr_decay=True, max_steps=max_step)
    self.assertEqual(len(result1), len(result2) / mb)
    if mb > 1:
      self.assertTrue(opt.get_slot_names() == ['accum', 'm', 'v'])
    ga_result = [sum(x) / mb for x in
                 np.array_split(result2, max_step)]
    result1 = [int(x * 1e6) / 1e6 for x in result1]
    ga_result = [int(x * 1e6) / 1e6 for x in ga_result]
    for r1, r2 in list(zip(result1, ga_result)):
      self.assertTrue(abs(r1-r2) <= 1.1e-6,
                      "{} and {} not equal, mb: {}, apply: {}" \
                      .format(result1, ga_result, mb, apply_num))

  def test_gradient_accumulation(self):
    for mb in [1, 4]:
      for apply_num in [1, 2]:
        self.compare_gradient_accumulation(mb, apply_num)

# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  test.main()
