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
"""Test for estimator loss."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

import tensorflow as tf
from tensorflow.python.platform import test

import epl

from estimator_test import capture_log
from estimator_test import input_fn
from estimator_test import model_fn
from test_utils import fix_randomness


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class EstimatorTest(test.TestCase):
  def test_compare_with_mirrorstrategy(self):
    fix_randomness()
    log_stream = capture_log()
    model_dir = tempfile.mkdtemp()
    max_steps = 10
    seed = 123123
    batch = 2
    distribution_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
    run_config = tf.estimator.RunConfig(train_distribute=distribution_strategy, tf_random_seed=seed,
                                        log_step_count_steps=1)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    estimator.train(lambda: input_fn(batch), max_steps=max_steps)
    logs = log_stream.getvalue()
    losses = [float(line.replace(',', ' ').split()[2]) for line in logs.strip().split('\n') if 'loss =' in line]
    res1 = [line.split(':')[-1] for line in logs.strip().split('\n') if 'loss =' in line]
    losses = [float(r.split(',')[0].split()[2]) for r in res1]
    shutil.rmtree(model_dir)
    log_stream.truncate(0)

    fix_randomness()

    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    model_dir = tempfile.mkdtemp()
    run_config = tf.estimator.RunConfig(tf_random_seed=seed, log_step_count_steps=1)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    estimator.train(lambda: input_fn(batch, False), max_steps=max_steps)
    logs = log_stream.getvalue()
    res2 = [line.split(':')[-1] for line in logs.strip().split('\n') if 'loss =' in line]
    losses2 = [float(r.split(',')[0].split()[2]) for r in res2]

    self.assertEqual(len(losses), len(losses2))
    for i, loss in enumerate(losses):
      self.assertTrue(abs(loss - losses2[i]) < 1e-6)
    shutil.rmtree(model_dir)

  def test_ignore_mirrorstrategy(self):
    log_stream = capture_log()
    model_dir = tempfile.mkdtemp()
    max_steps = 10
    seed = 123123
    batch = 2
    fix_randomness()
    distribution_strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=2)
    run_config = tf.estimator.RunConfig(train_distribute=distribution_strategy, tf_random_seed=seed,
                                        log_step_count_steps=1)
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)
    estimator.train(lambda: input_fn(batch), max_steps=max_steps)
    logs = log_stream.getvalue()
    losses = [float(line.replace(',', ' ').split()[2]) for line in logs.strip().split('\n') if 'loss =' in line]
    res1 = [line.split(':')[-1] for line in logs.strip().split('\n') if 'loss =' in line]


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  test.main()
