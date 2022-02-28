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
"""Test for MemoryProfilerHook."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.profiler.memory_profiler_hook import profile_memory
from epl.profiler.memory_profiler_hook import MemoryProfilerHook


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class MemoryProfilerHookTest(test.TestCase):
  def _model_def(self, ft=16):
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=ft, activation=None)
    x = tf.layers.dense(inputs=x, units=ft, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=ft, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return [loss, train_op, global_step]

  def test_hook(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    train_opts = self._model_def()
    max_steps = 10
    hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
    d = tempfile.mkdtemp()
    hooks.append(MemoryProfilerHook(save_steps=2, max_steps=4, output_dir=d,
                                    dump_metadata=True, visualize=False))
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      while not sess.should_stop():
        train_loss, _, step = sess.run(train_opts)
    r = ['run_metadata_4.bin', \
         'run_metadata_2.bin']
    self.assertTrue(sorted(r) == sorted(os.listdir(d)), os.listdir(d))
    shutil.rmtree(d)

  def test_peak(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    train_opts = self._model_def(1024)
    max_steps = 10
    hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
    run_option = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
      run_metadata = tf.RunMetadata()
      sess.run(train_opts, options=run_option,
               run_metadata=run_metadata)
      prefix_name = None
      stats = profile_memory(run_metadata, prefix_name, visualize=True)
      self.assertTrue(len(stats) == 2)
      print(stats)
      for st in stats:
        if 'GPU' in st['device']:
          self.assertTrue(st['device'] == \
                          '/job:worker/replica:0/task:0/device:GPU:0')
          self.assertTrue(abs(st['persist'] - 16967680) / st['persist'] < 0.2)
          peak = st['peak_memory']
          self.assertTrue(abs(peak - 29787136) / peak < 0.2)
        elif 'CPU' in st['device']:
          self.assertTrue(st['device'] == \
                          '/job:worker/replica:0/task:0/device:CPU:0')
          self.assertTrue(abs(st['persist']) == 0)


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  test.main()
