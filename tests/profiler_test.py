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
"""Test for Profiler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.profiler.profiler import profile_flops, profile_memory


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class ProfilerTest(test.TestCase):
  def _model_def(self):
    # dense1/MatMul
    # Tensor("IteratorGetNext:0", shape=(10, 10), dtype=float32)
    # Tensor("dense1/kernel/read:0", shape=(10, 16), dtype=float32)
    # dense1/BiasAdd
    # Tensor("dense1/MatMul:0", shape=(10, 16), dtype=float32)
    # Tensor("dense1/bias/read:0", shape=(16,), dtype=float32)
    # dense2/MatMul
    # Tensor("dense1/BiasAdd:0", shape=(10, 16), dtype=float32)
    # Tensor("dense2/kernel/read:0", shape=(16, 16), dtype=float32)
    # dense2/BiasAdd
    # Tensor("dense2/MatMul:0", shape=(10, 16), dtype=float32)
    # Tensor("dense2/bias/read:0", shape=(16,), dtype=float32)
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None, name="dense1")
    x = tf.layers.dense(inputs=x, units=16, activation=None, name="dense2")
    return x

  def test_profile_flops(self):
    epl.init()
    with tf.Graph().as_default():
      self._model_def()
      res = profile_flops()
      self.assertEqual(res["dense1/MatMul"], 10 * 10 * 16 * 2)
      self.assertEqual(res["dense2/MatMul"], 10 * 16 * 16 * 2)
      self.assertEqual(res["dense1/BiasAdd"], 10 * 16)
      self.assertEqual(res["dense2/BiasAdd"], 10 * 16)

  def test_profile_memory(self):
    epl.init()
    graph = tf.Graph()
    with graph.as_default():
      self._model_def()
      res = profile_memory()
      self.assertEqual(res["dense1/MatMul:0"], 10 * 16 * 4)
      self.assertEqual(res["dense2/MatMul:0"], 10 * 16 * 4)


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  test.main()
