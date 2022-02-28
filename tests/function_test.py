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
"""Test for function in taskgraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.utils import common
from epl.cluster import Cluster
from epl.ir.graph import Graph
from epl.utils.common import get_device_string

# pylint: disable=missing-docstring,protected-access,unused-argument,line-too-long,bad-continuation
_GPU_PER_WORKER = 4


def _mock_available_gpus():
  def available_gpus(self, *args, **kwargs):
    devices = []
    for gpu_index in range(_GPU_PER_WORKER):
      devices.append(get_device_string(task=0, device_index=gpu_index))
    return devices

  return available_gpus


Cluster.available_gpus = _mock_available_gpus()


class FunctionTest(test.TestCase):
  """Test import functions of parallelism transformation"""
  def _models_with_one_shot_iterator(self):
    with epl.replicate(device_count=1, name="stage_0"):
      num_x = np.random.randint(0, 10, (500, 20)).astype(dtype=np.float32)
      num_y = np.random.randint(0, 10, 500).astype(dtype=np.int64)
      dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
                               .batch(10).repeat(1)
      iterator = dataset.make_one_shot_iterator()
      x, labels = iterator.get_next()

      logits = tf.layers.dense(x, 2)
    with epl.replicate(device_count=1, name="stage_1"):
      logits = tf.layers.dense(logits, 10)
      loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                    logits=logits)
      return loss

  def test_cloned_one_shot_iterator(self):
    config = epl.Config()
    config.pipeline.num_micro_batch = 2
    epl.init(config)
    with tf.Graph().as_default():
      loss = self._models_with_one_shot_iterator()
      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      gvs = optimizer.compute_gradients(loss)
      optimizer.apply_gradients(gvs)
      g = Graph.get()
      self.assertFalse(g.clone_dataset_related_ops)
      g.clone_dataset_related_ops = True
      self.assertTrue(g.clone_dataset_related_ops)
      tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
          log_device_placement=False))
      functions = [function[:-12] for function in g.functions]
      list.sort(functions)
      real_functions = ["_make_dataset", \
                        "EPL_REPLICA_1/_make_dataset"]
      list.sort(real_functions)
      self.assertListEqual(functions, real_functions)
      real_nodes = \
          ["RepeatDataset", "RepeatDataset/count", \
           "BatchDatasetV2/drop_remainder", "BatchDatasetV2", \
           "BatchDatasetV2/batch_size", "TensorSliceDataset", \
           "TensorSliceDataset/tensors/component_0", \
           "TensorSliceDataset/tensors/component_1"]
      list.sort(real_nodes)
      for function in list(g.functions.values()):
        self.assertTrue(function.is_dataset_related)
        nodes = [node.name for node in list(function.nodes)]
        list.sort(nodes)
        self.assertTrue("_make_dataset" in function.name)
        self.assertTrue(nodes, real_nodes)

      taskgraph_0_functions = \
          [function.name[:-12] for function in g.taskgraphs[0].functions]
      list.sort(taskgraph_0_functions)
      self.assertListEqual(taskgraph_0_functions, real_functions)
      self.assertListEqual(
          [], [function.name[:-12] for function in g.taskgraphs[1].functions])

      for op, attr in list(g.op_with_function_map.items()):
        self.assertEqual(attr, "dataset_factory")
        self.assertEqual(op.type, "OneShotIterator")
        if common.get_replica_index_from_node_name(op.name):
          self.assertEqual(
              op.node_def.attr.get(attr).func.name[:-12],
              "EPL_REPLICA_1/_make_dataset")
        else:
          self.assertEqual(
              op.node_def.attr.get(attr).func.name[:-12], "_make_dataset")
      constructor_functions = [function for key, function in g.functions.items() if not key.startswith("EPL_REPLICA")]
      for func in constructor_functions:
        for node in func.nodes:
          self.assertTrue(node.device.endswith("device:CPU:0"))


# pylint: enable=missing-docstring,protected-access,unused-argument,
#                line-too-long,bad-continuation

if __name__ == "__main__":
  test.main()
