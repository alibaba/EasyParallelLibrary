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
"""Test for graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test

import epl
from epl.cluster import Cluster
from epl.ir.graph import Graph
from epl.ir.phase import ModelPhase
from epl.strategies.scheduler import get_scheduler
from epl.parallel.graph_editor import Custom
from epl.utils.common import get_device_string
from epl.utils import constant

# pylint: disable=missing-docstring,protected-access,unused-argument,too-many-nested-blocks,line-too-long,bad-continuation
_GPU_PER_WORKER = 8


def _mock_available_gpus():
  def available_gpus(self, *args, **kwargs):
    devices = []
    for gpu_index in range(_GPU_PER_WORKER):
      devices.append(get_device_string(task=0, device_index=gpu_index))
    return devices

  return available_gpus


Cluster.available_gpus = _mock_available_gpus()


class SchedulerTest(test.TestCase):
  """Test pipeline scheduler of parallelism transformation"""
  def _model_def(self):
    with epl.replicate(device_count=1, name='stage_0'):
      num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
      num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
      dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
          .batch(10).repeat(1)
      iterator = dataset.make_initializable_iterator()
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           iterator.initializer)
      x, _ = iterator.get_next()
      dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    with epl.replicate(device_count=1, name='stage_1'):
      dense2 = tf.layers.dense(inputs=dense1, units=10, activation=None)
    with epl.replicate(device_count=1, name='stage_2'):
      dense3 = tf.layers.dense(inputs=dense2, units=10, activation=None)
    with epl.replicate(device_count=1, name='stage_3'):
      logits = tf.layers.dense(inputs=dense3, units=10, activation=None)
      return tf.reduce_mean(logits)

  def test_scheduler(self):
    config = epl.Config()
    config.pipeline.num_micro_batch = 6
    config.pipeline.strategy = "PreferBackwardOptimizer"
    epl.init(config)
    with tf.Graph().as_default():
      loss = self._model_def()
      g = Graph.get()
      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)

      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      gvs = optimizer.compute_gradients(loss)
      optimizer.apply_gradients(gvs)

      tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
          log_device_placement=False))
      self.assertEqual(len(g.taskgraphs), 4)

      # Check pipeline config.
      pipeline_config = g.get_pipeline_config()
      self.assertEqual(pipeline_config.num_micro_batch, 6)
      self.assertEqual(pipeline_config.strategy, "PreferBackwardOptimizer")
      self.assertEqual(constant.DEFAUT_PIPELINE_STRATEGY, "PreferBackward")

      # Check num_stages.
      self.assertEqual(g.num_stages, 4)

      # Check scheduler of pipeline.
      scheduler = get_scheduler(pipeline_config.strategy)(pipeline_config.num_micro_batch, g.num_stages)
      # Clarify a custom obj for every .
      customs = []
      for taskgraph in g.taskgraphs:
        customs.append(Custom(taskgraph, 0))
      scheduler.call(customs)

      for micro_batch_idx in range(1, 6):
        for stage_idx in range(4):
          if stage_idx == 3:
            cur_dep_ops = customs[-1].backward_exit_ops[micro_batch_idx - 1]
            for orig_op in customs[-1].forward_entrance_ops[micro_batch_idx]:
              control_inputs = orig_op.control_inputs
              for ele in cur_dep_ops:
                self.assertTrue(ele.primitive_obj in control_inputs)
          else:
            forward_cache_num = 4
            if micro_batch_idx + stage_idx <= forward_cache_num:
              for orig_op in customs[stage_idx].forward_entrance_ops[
                  micro_batch_idx]:
                control_inputs = orig_op.control_inputs
                for ele in customs[stage_idx].forward_exit_ops[micro_batch_idx
                                                               - 1]:
                  self.assertTrue(ele.primitive_obj in control_inputs)
            else:
              reverse_idx = stage_idx + micro_batch_idx - 5
              for orig_op in customs[stage_idx].forward_entrance_ops[
                  micro_batch_idx]:
                control_inputs = orig_op.control_inputs
                for ele in customs[stage_idx].backward_exit_ops[reverse_idx]:
                  self.assertTrue(ele.primitive_obj in control_inputs)


# pylint: enable=missing-docstring,protected-access,unused-argument,too-many-nested-blocks,
#                line-too-long,bad-continuation

if __name__ == '__main__':
  test.main()
