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
"""Test for estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile
import numpy as np

import tensorflow as tf

import epl
from test_utils import fix_randomness


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
fix_randomness()
def model_fn(features, labels, mode):
  """Model function."""
  with tf.variable_scope('lr_softmax'):
    weights = tf.get_variable('weights', initializer=tf.zeros([78, 10]))
    biases = tf.get_variable('biases', initializer=tf.zeros([10]))
    logits = tf.matmul(features, weights) + biases
  global_step = tf.train.get_or_create_global_step()
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                logits=logits,
                                                weights=1.0)
  if mode == tf.estimator.ModeKeys.TRAIN:
    opt = tf.train.AdamOptimizer(0.1, name='adam')
    train_op = opt.minimize(loss, global_step=global_step, name='train')
    if epl.env.Env.get().cluster:
      epl.add_to_collection(loss, epl.GraphKeys.GLOBAL_MEAN_OBJECTS)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op)
  elif mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss)
  else:
    raise ValueError(
        "Only TRAIN and EVAL modes are supported: %s" % (mode))


def model_fn_replicate(features, labels, mode):
  with epl.replicate(device_count=1):
    return model_fn(features, labels, mode)

def input_fn(batch_size=2, estimator=True):
  """input function."""
  num_x = np.random.randint(0, 10, (200, 78)).astype(dtype=np.float32)
  num_y = np.random.randint(0, 10, 200).astype(dtype=np.int32)
  dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
      .batch(batch_size).repeat(10)
  if not estimator:
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  return dataset

def _main():
  config = epl.Config()
  epl.init(config)
  model_dir = tempfile.mkdtemp()
  model_dir = os.path.join(model_dir, str(epl.Env.get().cluster.worker_index))
  run_config = tf.estimator.RunConfig(save_checkpoints_steps=5,
                                      model_dir=model_dir)
  estimator = tf.estimator.Estimator(model_fn=model_fn_replicate, config=run_config)
  hooks = []
  hooks.append(tf.train.StepCounterHook(every_n_steps=1))
  train_spec = tf.estimator.TrainSpec(input_fn=input_fn, max_steps=30,
                                      hooks=hooks)
  eval_spec = tf.estimator.EvalSpec(input_fn=input_fn, steps=10)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
  shutil.rmtree(model_dir)
# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  _main()
