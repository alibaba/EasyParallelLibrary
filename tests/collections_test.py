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
"""Test for hook of session run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.ir.graph import Graph
from epl.ir.graph import GraphKeys


# pylint: disable=missing-docstring,unused-argument,unused-variable,line-too-long
class CollectionsTest(test.TestCase):
  def test_for_merged_output_map(self):
    epl.init(config=epl.Config({"communication.gradients_reduce_method": "sum"}))
    with epl.Cluster(worker_hosts="127.0.0.1:8001", worker_index=0):
      with epl.replicate(device_count=1):
        num_x = np.random.randint(0, 10, (500, 20)).astype(dtype=np.float32)
        num_y = np.random.randint(0, 10, 500).astype(dtype=np.int64)
        dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
                                .batch(10).repeat(1)
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                             iterator.initializer)
        x, labels = iterator.get_next()
        epl.add_to_collection([x, labels], epl.GraphKeys.GLOBAL_CONCAT_OBJECTS)

        logits = tf.layers.dense(x, 2)
        logits = tf.layers.dense(logits, 10)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)
        epl.add_to_collection(loss, epl.GraphKeys.GLOBAL_MEAN_OBJECTS)
        accuracy_1 = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, 1), tf.to_int64(labels)),
                    tf.float32))
        epl.add_to_collection(accuracy_1, epl.GraphKeys.GLOBAL_MEAN_OBJECTS)
        _, accuracy_2 = tf.metrics.accuracy(labels=tf.to_int64(labels),
                                            predictions=tf.argmax(logits, 1))
        epl.add_to_collection(accuracy_2, epl.GraphKeys.GLOBAL_MEAN_OBJECTS)

        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,
                                               momentum=0.9)
        train_op = optimizer.minimize(loss, global_step=global_step)
      tf.train.MonitoredTrainingSession()

      g = Graph.get()
      # check graphkeys
      self.assertEqual(GraphKeys.ALL_COLLECTION_KEYS, [
          GraphKeys.GLOBAL_CONCAT_OBJECTS, GraphKeys.GLOBAL_MEAN_OBJECTS,
          GraphKeys.GLOBAL_SUM_OBJECTS, GraphKeys.LOCAL_CONCAT_OBJECTS,
          GraphKeys.LOCAL_MEAN_OBJECTS, GraphKeys.LOCAL_SUM_OBJECTS
      ])

      # check collection
      global_allgather = [obj.name for obj in \
                          g.get_collection(GraphKeys.GLOBAL_CONCAT_OBJECTS)]
      global_mean = [obj.name for obj in \
                     g.get_collection(GraphKeys.GLOBAL_MEAN_OBJECTS)]
      for key, values in list(g.merged_outputs_map.items()):
        if not isinstance(values, list):
          g.merged_outputs_map[key] = values.name
          continue
        for v_idx, value in enumerate(values):
          g.merged_outputs_map[key][v_idx] = value.name

      self.assertListEqual(global_allgather,
                           ["IteratorGetNext:0", "IteratorGetNext:1"])
      self.assertListEqual(global_mean,
                           ["sparse_softmax_cross_entropy_loss/value:0", \
                            "Mean:0", "accuracy/update_op:0"])
      self.assertListEqual(g.get_collection(GraphKeys.GLOBAL_SUM_OBJECTS), [])
      self.assertListEqual(g.get_collection(GraphKeys.LOCAL_CONCAT_OBJECTS),
                           [])
      self.assertListEqual(g.get_collection(GraphKeys.LOCAL_MEAN_OBJECTS), [])
      self.assertListEqual(g.get_collection(GraphKeys.LOCAL_SUM_OBJECTS), [])
      self.assertDictEqual(
          g.merged_outputs_map,
          {"IteratorGetNext:0": \
               "EPL_PARALLEL_STRATEGY/Comm_0_allgather/0_1/EplNcclCommunicatorAllGather:0", \
           "IteratorGetNext:1": \
               "EPL_PARALLEL_STRATEGY/Comm_0_allgather/0_2/EplNcclCommunicatorAllGather:0", \
           "IteratorGetNext:1_replicated": \
               ["EPL_PARALLEL_STRATEGY/Comm_0_allgather/1_2/EplNcclCommunicatorAllGather:0", \
                "EPL_PARALLEL_STRATEGY/Comm_0_allgather/2_2/EplNcclCommunicatorAllGather:0"], \
           "sparse_softmax_cross_entropy_loss/value:0": "EPL_PARALLEL_STRATEGY/truediv:0", \
           "accuracy/update_op:0_replicated": \
               ["EPL_PARALLEL_STRATEGY/truediv_5:0", \
                "EPL_PARALLEL_STRATEGY/truediv_8:0"], \
           "Mean:0_replicated": \
               ["EPL_PARALLEL_STRATEGY/truediv_4:0", \
                "EPL_PARALLEL_STRATEGY/truediv_7:0"], \
           "Mean:0": "EPL_PARALLEL_STRATEGY/truediv_1:0", \
           "sparse_softmax_cross_entropy_loss/value:0_replicated": \
               ["EPL_PARALLEL_STRATEGY/truediv_3:0", \
                "EPL_PARALLEL_STRATEGY/truediv_6:0"], \
           "accuracy/update_op:0": "EPL_PARALLEL_STRATEGY/truediv_2:0", \
           "IteratorGetNext:0_replicated": \
               ["EPL_PARALLEL_STRATEGY/Comm_0_allgather/1_1/EplNcclCommunicatorAllGather:0", \
                "EPL_PARALLEL_STRATEGY/Comm_0_allgather/2_1/EplNcclCommunicatorAllGather:0"]})


# pylint: enable=missing-docstring,unused-argument,unused-variable,line-too-long

if __name__ == "__main__":
  test.main()
