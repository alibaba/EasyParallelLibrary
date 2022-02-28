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
"""Test for Hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from distutils.version import LooseVersion as Version
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework.versions import __version__

import epl

new_tf_version = \
    Version(__version__) < Version("2.0") and \
    Version(__version__) >= Version("1.14.0")


# pylint: disable=missing-docstring,protected-access,unused-argument
# pylint: disable=line-too-long,bad-continuation,unused-variable
class HookTest(test.TestCase):
  def test_loss(self):
    config = epl.Config()
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    labels = tf.ones([1], dtype=tf.int32)
    logits = tf.ones([1, 1], dtype=tf.float32)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits,
                                                  weights=1.0)
    groups = epl.Graph.get().group2ops
    op2group = epl.Graph.get().op2group
    self.assertEqual(len(groups), 1)
    self.assertEqual(list(groups.keys())[0], "sparse_softmax_cross_entropy_1")

    op2group_keys = list(op2group.keys())
    list.sort(op2group_keys)
    if new_tf_version:
      baseline = ["sparse_softmax_cross_entropy_loss/Const", \
                  "sparse_softmax_cross_entropy_loss/Const_1", \
                  "sparse_softmax_cross_entropy_loss/Const_2", \
                  "sparse_softmax_cross_entropy_loss/Mul", \
                  "sparse_softmax_cross_entropy_loss/Sum", \
                  "sparse_softmax_cross_entropy_loss/Sum_1", \
                  "sparse_softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success", \
                  "sparse_softmax_cross_entropy_loss/assert_broadcastable/values/rank", \
                  "sparse_softmax_cross_entropy_loss/assert_broadcastable/values/shape", \
                  "sparse_softmax_cross_entropy_loss/assert_broadcastable/weights/rank", \
                  "sparse_softmax_cross_entropy_loss/assert_broadcastable/weights/shape", \
                  "sparse_softmax_cross_entropy_loss/num_present", \
                  "sparse_softmax_cross_entropy_loss/num_present/Const", \
                  "sparse_softmax_cross_entropy_loss/num_present/Equal", \
                  "sparse_softmax_cross_entropy_loss/num_present/Equal/y", \
                  "sparse_softmax_cross_entropy_loss/num_present/Select", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rank", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shape", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Shape", \
                  "sparse_softmax_cross_entropy_loss/num_present/ones_like", \
                  "sparse_softmax_cross_entropy_loss/num_present/ones_like/Const", \
                  "sparse_softmax_cross_entropy_loss/num_present/ones_like/Shape", \
                  "sparse_softmax_cross_entropy_loss/num_present/zeros_like", \
                  "sparse_softmax_cross_entropy_loss/value", \
                  "sparse_softmax_cross_entropy_loss/xentropy/Shape", \
                  "sparse_softmax_cross_entropy_loss/xentropy/xentropy"]

    else:
      baseline = ["sparse_softmax_cross_entropy_loss/Const", \
                  "sparse_softmax_cross_entropy_loss/Const_1", \
                  "sparse_softmax_cross_entropy_loss/Const_2", \
                  "sparse_softmax_cross_entropy_loss/Equal", \
                  "sparse_softmax_cross_entropy_loss/Equal/y", \
                  "sparse_softmax_cross_entropy_loss/Greater", \
                  "sparse_softmax_cross_entropy_loss/Greater/y", \
                  "sparse_softmax_cross_entropy_loss/Mul", \
                  "sparse_softmax_cross_entropy_loss/Select", \
                  "sparse_softmax_cross_entropy_loss/Sum", \
                  "sparse_softmax_cross_entropy_loss/Sum_1", \
                  "sparse_softmax_cross_entropy_loss/assert_broadcastable/static_scalar_check_success", \
                  "sparse_softmax_cross_entropy_loss/assert_broadcastable/values/rank", \
                  "sparse_softmax_cross_entropy_loss/assert_broadcastable/values/shape", \
                  "sparse_softmax_cross_entropy_loss/assert_broadcastable/weights/rank", \
                  "sparse_softmax_cross_entropy_loss/assert_broadcastable/weights/shape", \
                  "sparse_softmax_cross_entropy_loss/div", \
                  "sparse_softmax_cross_entropy_loss/num_present", \
                  "sparse_softmax_cross_entropy_loss/num_present/Const", \
                  "sparse_softmax_cross_entropy_loss/num_present/Equal", \
                  "sparse_softmax_cross_entropy_loss/num_present/Equal/y", \
                  "sparse_softmax_cross_entropy_loss/num_present/Select", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/static_scalar_check_success", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/rank", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/values/shape", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/rank", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/assert_broadcastable/weights/shape", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Const", \
                  "sparse_softmax_cross_entropy_loss/num_present/broadcast_weights/ones_like/Shape", \
                  "sparse_softmax_cross_entropy_loss/num_present/ones_like", \
                  "sparse_softmax_cross_entropy_loss/num_present/ones_like/Const", \
                  "sparse_softmax_cross_entropy_loss/num_present/ones_like/Shape", \
                  "sparse_softmax_cross_entropy_loss/num_present/zeros_like", \
                  "sparse_softmax_cross_entropy_loss/ones_like", \
                  "sparse_softmax_cross_entropy_loss/ones_like/Const", \
                  "sparse_softmax_cross_entropy_loss/ones_like/Shape", \
                  "sparse_softmax_cross_entropy_loss/value", \
                  "sparse_softmax_cross_entropy_loss/xentropy/Shape", \
                  "sparse_softmax_cross_entropy_loss/xentropy/xentropy", \
                  "sparse_softmax_cross_entropy_loss/zeros_like"]

    self.assertEqual(baseline, op2group_keys)

  def test_nn_dropout(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    x = tf.ones([1, 1], dtype=tf.float32)
    x = tf.nn.dropout(x, 0.5)
    groups = epl.Graph.get().group2ops
    op2group = epl.Graph.get().op2group
    self.assertEqual(len(groups), 1)
    self.assertEqual(list(groups.keys())[0], "dropout_1")
    op2group_keys = list(op2group.keys())
    list.sort(op2group_keys)

    print(op2group_keys)

    if new_tf_version:
      baseline = ["dropout/Cast", \
                  "dropout/GreaterEqual", \
                  "dropout/Shape", \
                  "dropout/mul", \
                  "dropout/mul_1", \
                  "dropout/random_uniform", \
                  "dropout/random_uniform/RandomUniform", \
                  "dropout/random_uniform/max", \
                  "dropout/random_uniform/min", \
                  "dropout/random_uniform/mul", \
                  "dropout/random_uniform/sub", \
                  "dropout/rate", \
                  "dropout/sub", \
                  "dropout/sub/x", \
                  "dropout/truediv", \
                  "dropout/truediv/x"]
    else:
      baseline = ["dropout/Floor", \
                  "dropout/Shape", \
                  "dropout/add", \
                  "dropout/div", \
                  "dropout/keep_prob", \
                  "dropout/mul", \
                  "dropout/random_uniform", \
                  "dropout/random_uniform/RandomUniform", \
                  "dropout/random_uniform/max", \
                  "dropout/random_uniform/min", \
                  "dropout/random_uniform/mul", \
                  "dropout/random_uniform/sub"]
    self.assertEqual(baseline, op2group_keys)

  def test_nn_rnn(self):
    config = epl.Config()
    epl.init(config)
    epl.set_default_strategy(epl.replicate(1))
    tf.nn.rnn_cell.LSTMCell(num_units=5,
                            num_proj=2,
                            use_peepholes=True,
                            cell_clip=50,
                            forget_bias=0.0)
    groups = epl.Graph.get().group2ops
    op2group = epl.Graph.get().op2group
    self.assertEqual(len(groups), 0)
    self.assertEqual(len(op2group), 0)
# pylint: disable=missing-docstring,protected-access,unused-argument
# pylint: disable=line-too-long,bad-continuation,unused-variable

if __name__ == "__main__":
  test.main()
