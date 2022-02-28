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
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework.versions import __version__
import numpy as np

import epl
from epl.parallel.planner import AutoStageGenerator
from epl.parallel.partitioner import partition_buckets
from epl.utils.constant import AUTO_STAGE_POLICY_BALANCE_OP_NUM


# pylint: disable=missing-docstring,unused-argument,unused-variable
# pylint: disable=protected-access
class PlannerTest(test.TestCase):
  def _model_def(self):
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    return tf.reduce_mean(logits)

  def test_partition_buckets(self):
    weights = [1, 1, 1, 1, 1, 1]
    parts = partition_buckets(weights, 2, 3)
    self.assertEqual(parts, [(0, 2), (2, 2), (4, 2)])
    parts = partition_buckets(weights, 3, 3)
    self.assertEqual(parts, [(0, 3), (3, 3)])

  def test_partition_buckets2(self):
    weights = [1, 7, 4, 3, 1, 11]
    parts = partition_buckets(weights, 1, 3)
    self.assertEqual(parts, None)
    parts = partition_buckets(weights, 5, 3)
    self.assertEqual(parts, None)
    parts = partition_buckets(weights, 11, 3)
    self.assertEqual(parts, [(0, 8), (2, 8), (5, 11)])

  def test_search_op_num(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    plan = AutoStageGenerator(policy=AUTO_STAGE_POLICY_BALANCE_OP_NUM, num_stages=2)
    model = self._model_def()
    stage_ops = plan.search()
    # total 46 ops
    new_tf_version = \
        Version(__version__) < Version("2.0") and \
        Version(__version__) >= Version("1.14.0")
    self.assertEqual(len(stage_ops), 2)
    self.assertEqual(len(stage_ops[0]), 25 if new_tf_version else 23)
    self.assertEqual(len(stage_ops[1]), 24 if new_tf_version else 23)

  def test_search_op_num2(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    plan = AutoStageGenerator(policy=AUTO_STAGE_POLICY_BALANCE_OP_NUM, num_stages=3)
    model = self._model_def()
    stage_ops = plan.search()
    # total 46 ops
    new_tf_version = \
        Version(__version__) < Version("2.0") and \
        Version(__version__) >= Version("1.14.0")
    self.assertEqual(len(stage_ops), 3)
    self.assertEqual(len(stage_ops[0]), 17 if new_tf_version else 16)
    self.assertEqual(len(stage_ops[1]), 17 if new_tf_version else 16)
    self.assertEqual(len(stage_ops[2]), 15 if new_tf_version else 14)


# pylint: enable=missing-docstring,unused-argument,unused-variable

if __name__ == "__main__":
  test.main()
