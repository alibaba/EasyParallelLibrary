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
"""Test for summary merge."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.ir.graph import Graph


# pylint: disable=missing-docstring,unused-argument,unused-variable,line-too-long
class SummaryTest(test.TestCase):
  def test_for_summary(self):
    config = epl.Config()
    config.communication.gradients_reduce_method = "sum"
    config.pipeline.num_micro_batch = 3
    epl.init(config=epl.Config({"communication.gradients_reduce_method": "sum"}))
    with epl.replicate(device_count=1):
      num_x = np.random.randint(0, 10,
                                (500, 20)).astype(dtype=np.float32)
      num_y = np.random.randint(0, 10, 500).astype(dtype=np.int64)
      dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
                              .batch(10).repeat(1)
      iterator = dataset.make_initializable_iterator()
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           iterator.initializer)
      x, labels = iterator.get_next()
      epl.add_to_collection([x, labels], epl.GraphKeys.GLOBAL_CONCAT_OBJECTS)
      tf.summary.histogram("features", x)
      tf.summary.histogram("labels", labels)

      logits = tf.layers.dense(x, 2)
    with epl.replicate(device_count=1):
      logits = tf.layers.dense(logits, 10)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                    logits=logits)
      epl.add_to_collection(loss, epl.GraphKeys.GLOBAL_MEAN_OBJECTS)
      accuracy_1 = \
          tf.reduce_mean(tf.cast(tf.equal(
              tf.argmax(logits, 1), tf.to_int64(labels)), tf.float32))
      tf.summary.scalar("mean_acc", accuracy_1)
      epl.add_to_collection(accuracy_1, epl.GraphKeys.GLOBAL_MEAN_OBJECTS)
      _, accuracy_2 = tf.metrics.accuracy(labels=tf.to_int64(labels),
                                          predictions=tf.argmax(
                                              logits, 1))
      tf.summary.scalar("metrics_acc", accuracy_2)
      epl.add_to_collection(accuracy_2, epl.GraphKeys.GLOBAL_MEAN_OBJECTS)

    # check summary collections before update
    summary_list = tf.get_default_graph().get_collection(
        tf.GraphKeys.SUMMARIES)
    summary_list = [obj.name for obj in summary_list]
    self.assertListEqual(
        summary_list,
        ["features:0", "labels:0", "mean_acc:0", "metrics_acc:0"])

    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    tf.train.MonitoredTrainingSession()

    g = Graph.get()
    # check summary collections after update
    summary_list = tf.get_default_graph().get_collection(
        tf.GraphKeys.SUMMARIES)
    summary_list = [obj.name for obj in summary_list]
    self.assertListEqual(summary_list, [
        "EPL_PARALLEL_STRATEGY/features:0", "EPL_PARALLEL_STRATEGY/labels:0",
        "EPL_PARALLEL_STRATEGY/mean_acc:0",
        "EPL_PARALLEL_STRATEGY/metrics_acc:0"
    ])

    # check summary map
    summary_map_list = list(g.summary_map.keys())
    list.sort(summary_map_list)
    self.assertListEqual(summary_map_list, \
                         ["EPL_PARALLEL_STRATEGY/features:0", \
                         "EPL_PARALLEL_STRATEGY/labels:0", \
                         "EPL_PARALLEL_STRATEGY/mean_acc:0", \
                         "EPL_PARALLEL_STRATEGY/metrics_acc:0"])
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/features:0"].tags, \
        "features")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/features:0"].tensor_name, \
        "EPL_PARALLEL_STRATEGY/Comm_0_allgather/0_1/EplNcclCommunicatorAllGather:0")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/features:0"].summary_type, \
          "SUMMARY_HISTOGRAM_TYPE")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/labels:0"].tags, \
        "labels")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/labels:0"].tensor_name, \
        "EPL_PARALLEL_STRATEGY/Comm_0_allgather/0_2/EplNcclCommunicatorAllGather:0")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/labels:0"].summary_type, \
        "SUMMARY_HISTOGRAM_TYPE")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/metrics_acc:0"].tags, \
        "metrics_acc")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/metrics_acc:0"].tensor_name, \
        "EPL_PARALLEL_STRATEGY/truediv_2:0")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/metrics_acc:0"].summary_type, \
        "SUMMARY_SCALAR_TYPE")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/mean_acc:0"].tags, \
        "mean_acc")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/mean_acc:0"].tensor_name, \
        "EPL_PARALLEL_STRATEGY/truediv_1:0")
    self.assertEqual(
        g.summary_map["EPL_PARALLEL_STRATEGY/mean_acc:0"].summary_type, \
        "SUMMARY_SCALAR_TYPE")


# pylint: enable=missing-docstring,unused-argument,unused-variable,line-too-long
if __name__ == "__main__":
  test.main()
