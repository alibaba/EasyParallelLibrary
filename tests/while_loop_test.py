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
"""Test for while-loop."""

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
from epl.ir.phase import ModelPhase


# pylint: disable=missing-docstring,protected-access,unused-argument
# pylint: disable=line-too-long,bad-continuation,unused-variable
_GPU_PER_WORKER = 4


def _mock_available_gpus():
  def available_gpus(self, *args, **kwargs):
    devices = []
    for gpu_index in range(_GPU_PER_WORKER):
      devices.append(common.get_device_string(task=0, device_index=gpu_index))
    return devices

  return available_gpus


def input_to_tensorarray(value, axis, size=None):
  shape = value.get_shape().as_list()
  rank = len(shape)
  dtype = value.dtype
  array_size = shape[axis] if not shape[axis] is None else size

  if array_size is None:
    raise ValueError("Can't create TensorArray with size None")

  array = tf.TensorArray(dtype=dtype, size=array_size)
  dim_permutation = [axis] + list(range(1, axis)) + [0] + list(
      range(axis + 1, rank))
  unpack_axis_major_value = tf.transpose(value, dim_permutation)
  full_array = array.unstack(unpack_axis_major_value)
  return full_array


Cluster.available_gpus = _mock_available_gpus()


class WhileLoopTest(test.TestCase):
  """Test parallelism transformation of WhileLoop case."""
  def _model_def(self):
    with epl.replicate(device_count=1, name="stage_0"):
      num_x = np.random.randint(0, 10,
                                (500, 10, 10)).astype(dtype=np.float32)
      num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
      seq_len = 10
      hidden_dim = 10
      dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
          .batch(10).repeat(1)
      iterator = dataset.make_initializable_iterator()
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           iterator.initializer)
      x, y = iterator.get_next()
      u = tf.get_variable(name='U',
                          shape=[10, hidden_dim],
                          dtype=tf.float32)  # from input to hidden
      b_u = tf.get_variable(name='b_U',
                            shape=[hidden_dim],
                            dtype=tf.float32)

      v = tf.get_variable(name='V',
                          shape=[hidden_dim, 10],
                          dtype=tf.float32)  # from hidden to output
      b_v = tf.get_variable(name='b_V', shape=[10], dtype=tf.float32)

      w = tf.get_variable(name='W',
                          shape=[hidden_dim, hidden_dim],
                          dtype=tf.float32)  # from hidden to hidden
      b_w = tf.get_variable(name='b_W',
                            shape=[hidden_dim],
                            dtype=tf.float32)
      input_ta = input_to_tensorarray(x, 1, seq_len)
      h = tf.TensorArray(tf.float32, seq_len + 1, clear_after_read=False)
      h = h.write(0,
                  tf.constant(np.zeros((1, hidden_dim)), dtype=tf.float32))
      output = tf.TensorArray(tf.float32, seq_len)
      time = tf.constant(0, dtype=tf.int32)

      def loop_body(time, h, output):
        input_step = input_ta.read(time)
        h_prev = h.read(time)
        h = h.write(
            time + 1,
            tf.tanh(
                tf.matmul(input_step, u) + b_u + tf.matmul(h_prev, w) +
                b_w))
        output = output.write(time, tf.matmul(h.read(time + 1), v) + b_v)
        return (time + 1, h, output)

      # build graph using while_loop
      loop_cond_fn = lambda time, _1, _2: time < seq_len
      final_state_ = tf.while_loop(cond=loop_cond_fn,
                                   body=loop_body,
                                   loop_vars=(time, h, output))

      final_state = final_state_
      final_output = final_state[-1].read(seq_len - 1)
    with epl.replicate(device_count=1, name="stage_1"):
      loss = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                     logits=tf.reshape(
                                                         final_output,
                                                         shape=[10]))
      return tf.reduce_mean(loss)

  def test_while_loop(self):
    config = epl.Config()
    config.pipeline.num_micro_batch = 2
    epl.init(config)
    with tf.Graph().as_default():
      loss = self._model_def()

      g = Graph.get()
      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)

      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      gvs = optimizer.compute_gradients(loss)
      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)

      optimizer.apply_gradients(gvs)
      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)
      tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
          log_device_placement=False))
      self.assertEqual(len(g.taskgraphs), 2)

      # Check context.
      for op_name in g.operations:
        if common.get_micro_batch_index_from_node_name(op_name) or \
            common.get_replica_index_from_node_name(op_name):
          continue
        op = g.get_operation_by_name(op_name)
        for replica_idx in range(2):
          replica_prefix = common.get_replica_prefix(replica_idx)
          for micro_batch_idx in range(2):
            if not replica_idx and not micro_batch_idx:
              continue
            micro_batch_prefix = common.get_micro_batch_prefix(micro_batch_idx)
            cloned_op_name = replica_prefix + micro_batch_prefix + op_name
            if cloned_op_name not in g.operations:
              continue
            cloned_op = g.get_operation_by_name(cloned_op_name)
            if op.get_control_flow_context() is not None:
              old_context = op.get_control_flow_context().to_proto(
              ).context_name
              new_context = cloned_op.get_control_flow_context().to_proto(
              ).context_name
              self.assertEqual(
                  old_context,
                  common.get_original_name_from_cloned_object(new_context))
              self.assertEqual(
                  replica_idx,
                  common.get_replica_index_from_node_name(new_context))
              self.assertEqual(
                  micro_batch_idx,
                  common.get_micro_batch_index_from_node_name(new_context))
            for inp_idx, inp in enumerate(op.inputs):
              cloned_inp_name = \
                  replica_prefix + micro_batch_prefix + inp.name
              if cloned_inp_name in g.tensors:
                cloned_replica_idx = \
                    common.get_replica_index_from_node_name(
                        cloned_op.inputs[inp_idx].name)
                self.assertEqual(replica_idx, cloned_replica_idx)
                cloned_micro_batch_idx = \
                    common.get_micro_batch_index_from_node_name(
                        cloned_op.inputs[inp_idx].name)
                self.assertEqual(micro_batch_idx, cloned_micro_batch_idx)


# pylint: enable=missing-docstring,protected-access,unused-argument,
# pylint: enable=line-too-long,bad-continuation,unused-variable

if __name__ == "__main__":
  test.main()
