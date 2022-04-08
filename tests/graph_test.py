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

import copy
from distutils.version import LooseVersion as Version
import warnings
import numpy as np
import tensorflow as tf
from tensorflow.contrib.data.python.ops import threadpool
from tensorflow.python.framework.versions import __version__
from tensorflow.python.platform import test

import epl
from epl.utils import constant
from epl.config import Config
from epl.env import Env
from epl.ir.graph import Graph
from epl.ir.graph import GraphKeys
from epl.ir.phase import ModelPhase
from epl.parallel.graph_editor import Custom

warnings.simplefilter("always")

# pylint: disable=missing-docstring,protected-access,unused-argument
# pylint: disable=line-too-long,bad-continuation,unused-variable
class GraphTest(test.TestCase):
  """Test import functions of parallelism transformation"""
  def _model_def(self):
    with epl.replicate(device_count=1, name="stage_0"):
      num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
      num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
      dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
          .batch(10).repeat(1)
      iterator = dataset.make_initializable_iterator()
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                           iterator.initializer)
      x, _ = iterator.get_next()
      dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    with epl.replicate(device_count=1, name="stage_1"):
      logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
      return tf.reduce_mean(logits)

  def test_graph_with_clip(self):
    config = epl.Config()
    config.pipeline.num_micro_batch = 2
    epl.init(config)
    with tf.Graph().as_default():
      loss = self._model_def()
      epl.add_to_collection(loss, GraphKeys.GLOBAL_MEAN_OBJECTS)
      g = Graph.get()

      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)

      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      tvars = tf.trainable_variables()
      grads = tf.gradients(loss, tvars)
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
      optimizer.apply_gradients(list(zip(grads, tvars)))
      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)
      tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
          log_device_placement=False))

      # check first taskgraph.
      b_exit_op_list = list(g.taskgraphs[0].backward_exit_ops(0, 0))
      for b_exit_op in b_exit_op_list:
        self.assertEqual(len(b_exit_op.outputs), 1)
        self.assertEqual(len(b_exit_op.outputs[0].consumers), 2)
        consumers_type = \
            [b_exit_op.type for b_exit_op in \
            list(b_exit_op.outputs[0].consumers)]
        self.assertTrue(("Add" in consumers_type)
                        ^ ("AddV2" in consumers_type))
      b_exit_op_list = [b_exit_op.name for b_exit_op in b_exit_op_list]
      list.sort(b_exit_op_list)
      self.assertEqual(b_exit_op_list, [
          "clip_by_global_norm/clip_by_global_norm/_0",
          "clip_by_global_norm/clip_by_global_norm/_1"
      ])

      # check second taskgraph.
      s_1_b_exit_0_0 = g.taskgraphs[1].backward_exit_ops(0, 0)
      for b_exit_op in s_1_b_exit_0_0:
        self.assertEqual(len(b_exit_op.outputs), 1)
        self.assertEqual(len(b_exit_op.outputs[0].consumers), 2)
        consumers_type = \
            [b_exit_op.type for b_exit_op in \
            list(b_exit_op.outputs[0].consumers)]
        self.assertTrue(("Add" in consumers_type)
                        ^ ("AddV2" in consumers_type))
      self.assertEqual(len(s_1_b_exit_0_0), 2)
      s_1_b_exit_0_0 = \
          [b_exit_op.name for b_exit_op in s_1_b_exit_0_0]
      list.sort(s_1_b_exit_0_0)
      self.assertEqual(s_1_b_exit_0_0, [
          "clip_by_global_norm/clip_by_global_norm/_2",
          "clip_by_global_norm/clip_by_global_norm/_3"
      ])

  def test_graph_with_clip_and_scale(self):
    config = epl.Config()
    config.pipeline.num_micro_batch = 2
    epl.init(config)
    with tf.Graph().as_default():
      loss = self._model_def()
      g = Graph.get()
      epl.add_to_collection(loss, GraphKeys.GLOBAL_MEAN_OBJECTS)

      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)

      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      tvars = tf.trainable_variables()
      grads = tf.gradients(loss, tvars)
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
      # Scale gradients manually
      grads = [grad * float(1 / 2) for grad in grads]
      optimizer.apply_gradients(list(zip(grads, tvars)))
      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)
      tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
          log_device_placement=False))

      # check first taskgraph.
      b_exit_op_list = list(g.taskgraphs[0].backward_exit_ops(0, 0))
      for b_exit_op in b_exit_op_list:
        self.assertEqual(len(b_exit_op.outputs), 1)
        self.assertEqual(len(b_exit_op.outputs[0].consumers), 2)
        consumers_type = \
            [b_exit_op.type for b_exit_op in \
            list(b_exit_op.outputs[0].consumers)]
        self.assertTrue(("Add" in consumers_type)
                        ^ ("AddV2" in consumers_type))
      b_exit_op_list = [b_exit_op.name for b_exit_op in b_exit_op_list]
      list.sort(b_exit_op_list)
      self.assertEqual(b_exit_op_list, ["mul", "mul_1"])

      # check second taskgraph.
      s_1_b_exit_0_0 = g.taskgraphs[1].backward_exit_ops(0, 0)
      for b_exit_op in s_1_b_exit_0_0:
        self.assertEqual(len(b_exit_op.outputs), 1)
        self.assertEqual(len(b_exit_op.outputs[0].consumers), 2)
        consumers_type = \
            [b_exit_op.type for b_exit_op in \
            list(b_exit_op.outputs[0].consumers)]
        self.assertTrue(("Add" in consumers_type)
                        ^ ("AddV2" in consumers_type))
      self.assertEqual(len(s_1_b_exit_0_0), 2)
      s_1_b_exit_0_0 = \
          [b_exit_op.name for b_exit_op in s_1_b_exit_0_0]
      list.sort(s_1_b_exit_0_0)
      self.assertEqual(s_1_b_exit_0_0, ["mul_2", "mul_3"])

  def test_graph_with_clip_after_allreduce(self):
    conf = Config()
    # Clip gradients after allreduce
    conf.communication.clip_after_allreduce = True
    conf.pipeline.num_micro_batch = 2
    epl.init(conf)
    with tf.Graph().as_default():
      loss = self._model_def()
      g = Graph.get()
      epl.add_to_collection(loss, GraphKeys.GLOBAL_MEAN_OBJECTS)

      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)

      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      tvars = tf.trainable_variables()
      grads = tf.gradients(loss, tvars)
      (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
      optimizer.apply_gradients(list(zip(grads, tvars)))
      self.assertEqual(g._current_model_phase, ModelPhase.FORWARD)
      tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
          log_device_placement=False))

      # check first taskgraph.
      b_exit_op_list = list(g.taskgraphs[0].backward_exit_ops(0, 0))
      b_exit_op_list = [b_exit_op.name for b_exit_op in b_exit_op_list]
      list.sort(b_exit_op_list)
      self.assertEqual(b_exit_op_list, [
          "gradients/dense/BiasAdd_grad/BiasAddGrad",
          "gradients/dense/MatMul_grad/MatMul_1"
      ])

      # check second taskgraph.
      s_1_b_exit_0_0 = g.taskgraphs[1].backward_exit_ops(0, 0)
      s_1_b_exit_0_0 = \
          [b_exit_op.name for b_exit_op in s_1_b_exit_0_0]
      list.sort(s_1_b_exit_0_0)
      self.assertEqual(s_1_b_exit_0_0, [
          "gradients/dense_1/BiasAdd_grad/BiasAddGrad",
          "gradients/dense_1/MatMul_grad/MatMul_1"
      ])

  def test_outside_strategy_error(self):
    epl.init()
    with tf.Graph().as_default():
      x1 = tf.constant(1.1, shape=[2, 2])
      with epl.replicate(name="replica_0"):
        x2 = tf.constant(1.1, shape=[2, 2])
      with epl.replicate(name="replica_1"):
        dense = tf.layers.dense(inputs=x2, units=2)
        with warnings.catch_warnings(record=True) as w:
          loss = tf.reduce_mean(dense)
        with warnings.catch_warnings(record=True) as w:
          optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        self.assertEqual(0, len(w))
        with warnings.catch_warnings(record=True) as w:
          gvs = optimizer.compute_gradients(loss)
        self.assertEqual(1, len(w))
        with warnings.catch_warnings(record=True) as w:
          optimizer.apply_gradients(gvs)
        self.assertEqual(1, len(w))

      x3 = tf.constant(1.1, shape=[2, 2])
      self.assertEqual(Graph.get()._user_default_taskgraph, None)
      self.assertNotEqual(Graph.get()._epl_default_taskgraph, None)
      self.assertEqual(Graph.get()._epl_default_taskgraph.index, 1)
      epl.set_default_strategy(epl.replicate(1, name="replica_2"))
      x4 = tf.constant(1.1, shape=[2, 2])
      with epl.replicate(name="replica_3"):
        x5 = tf.constant(1.1, shape=[2, 2])
      self.assertNotEqual(Graph.get()._epl_default_taskgraph, None)
      self.assertNotEqual(Graph.get()._user_default_taskgraph, None)
      self.assertEqual(Graph.get()._user_default_taskgraph.index, 2)
      self.assertEqual(Graph.get()._epl_default_taskgraph.index, 3)

  def test_graph(self):
    config = epl.Config()
    config.pipeline.num_micro_batch = 2
    epl.init(config)
    with tf.Graph().as_default():
      loss = self._model_def()
      epl.add_to_collection(loss, GraphKeys.GLOBAL_MEAN_OBJECTS)
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

      # Test traverse_depend_ops.
      depend_ops = g.traverse_depend_ops("IteratorGetNext",
                                         consider_outputs=False)
      self.assertEqual(len(depend_ops), 1)
      self.assertEqual(depend_ops[0].name, "IteratorV2")

      dataset_depend_ops = g.get_dataset_related_ops()
      if Version(__version__) < Version("1.14.0"):
        self.assertEqual(len(dataset_depend_ops), 11)
        assert_depend_ops = [
            "IteratorV2", "IteratorToStringHandle", "MakeIterator",
            "RepeatDataset", "BatchDatasetV2", "count", "TensorSliceDataset",
            "batch_size", "drop_remainder", "tensors/component_0",
            "tensors/component_1"
        ]

      elif Version(__version__) < Version("2.0"):
        assert_depend_ops = ([
            "TensorSliceDataset", "IteratorToStringHandle", "RepeatDataset",
            "BatchDatasetV2", "count", "ModelDataset", "OptimizeDataset",
            "IteratorV2", "batch_size", "normalize_element/component_0",
            "normalize_element/component_1", "drop_remainder", "MakeIterator",
            "optimizations"
        ])
      else:
        raise RuntimeError("Version of tensorflow is not supported for now."
                           "Tenosrflow Version: %s." % __version__)

      for op in assert_depend_ops:
        self.assertTrue(g.get_operation_by_name(op) in dataset_depend_ops)

      # check get_local_replicas
      obj = g.get_operation_by_name("IteratorV2")
      self.assertEqual(g.get_local_replicas(obj), [])
      obj = g.get_operation_by_name("dense/BiasAdd")
      local_replicas = [x.name for x in g.get_local_replicas(obj)]
      self.assertEqual(local_replicas, ["EPL_REPLICA_1/dense/BiasAdd"])
      obj = g.get_operation_by_name("EPL_REPLICA_1/dense/BiasAdd")
      local_replicas = [x.name for x in g.get_local_replicas(obj)]
      self.assertEqual(local_replicas, [])
      obj = g.get_operation_by_name("EPL_MICRO_BATCH_1/dense/BiasAdd")
      local_replicas = [x.name for x in g.get_local_replicas(obj)]
      self.assertEqual(local_replicas,
                       ["EPL_REPLICA_1/EPL_MICRO_BATCH_1/dense/BiasAdd"])
      # check get_local_micro_batches
      obj = g.get_operation_by_name("IteratorV2")
      self.assertEqual(g.get_local_micro_batches(obj), [])
      obj = g.get_operation_by_name("dense/BiasAdd")
      local_micro_batches = \
          [x.name for x in g.get_local_micro_batches(obj)]
      self.assertEqual(local_micro_batches,
                       ["EPL_MICRO_BATCH_1/dense/BiasAdd"])
      obj = g.get_operation_by_name("EPL_REPLICA_1/dense/BiasAdd")
      local_micro_batches = \
          [x.name for x in g.get_local_micro_batches(obj)]
      self.assertEqual(local_micro_batches,
                       ["EPL_REPLICA_1/EPL_MICRO_BATCH_1/dense/BiasAdd"])
      obj = g.get_operation_by_name("EPL_MICRO_BATCH_1/dense/BiasAdd")
      local_micro_batches = \
          [x.name for x in g.get_local_micro_batches(obj)]
      self.assertEqual(local_micro_batches, [])

      # check first taskgraph.
      self.assertTrue(g.taskgraphs[0].is_first_stage)
      self.assertTrue(g.pipeline_enabled)
      self.assertEqual(g.taskgraphs[0].pipeline_config.num_micro_batch, 2)
      self.assertEqual(g.taskgraphs[0].num_replicas, 2)
      self.assertEqual(g.taskgraphs[0].local_num_replicas, 2)
      broadcast_tensors_0 = \
          [tensor.name for tensor in g.taskgraphs[0].get_variables(0)]
      self.assertEqual(broadcast_tensors_0, [
          "dense/kernel:0", "dense/bias:0", "beta1_power:0", "beta2_power:0",
          "dense/kernel/Adam:0", "dense/kernel/Adam_1:0", "dense/bias/Adam:0",
          "dense/bias/Adam_1:0"
      ])
      broadcast_tensors_1 = \
          [tensor.name for tensor in g.taskgraphs[0].get_variables(1)]
      self.assertEqual(broadcast_tensors_1, [
          "EPL_REPLICA_1/dense/kernel:0", "EPL_REPLICA_1/dense/bias:0",
          "EPL_REPLICA_1/beta1_power:0", "EPL_REPLICA_1/beta2_power:0",
          "EPL_REPLICA_1/dense/kernel/Adam:0",
          "EPL_REPLICA_1/dense/kernel/Adam_1:0",
          "EPL_REPLICA_1/dense/bias/Adam:0",
          "EPL_REPLICA_1/dense/bias/Adam_1:0"
      ])

      gradients = [grad.name for grad in g.taskgraphs[0].gradients]
      list.sort(gradients)
      self.assertEqual(gradients, [
          "gradients/dense/BiasAdd_grad/tuple/control_dependency_1:0",
          "gradients/dense/MatMul_grad/tuple/control_dependency_1:0"
      ])
      self.assertEqual([
          f_ent_op.name
          for f_ent_op in list(g.taskgraphs[0].forward_entrance_ops(0, 0))
      ], ["IteratorGetNext"])
      self.assertEqual(len(g.taskgraphs[0].forward_exit_ops(0, 0)), 1)
      f_exit_ops = [f_exit_op.name for f_exit_op in \
                    list(g.taskgraphs[0].forward_exit_ops(0, 0))]
      list.sort(f_exit_ops)
      self.assertEqual(f_exit_ops, ["dense/BiasAdd"])
      self.assertEqual(len(g.taskgraphs[0].backward_entrance_ops(0, 0)), 2)
      b_ent_op_list = [
          b_ent_op.name
          for b_ent_op in g.taskgraphs[0].backward_entrance_ops(0, 0)
      ]
      list.sort(b_ent_op_list)
      self.assertEqual(b_ent_op_list, [
          "gradients/dense/MatMul_grad/MatMul",
          "gradients/dense/MatMul_grad/MatMul_1"
      ])
      b_exit_op_list = [
          b_exit_op.name
          for b_exit_op in list(g.taskgraphs[0].backward_exit_ops(0, 0))
      ]
      list.sort(b_exit_op_list)
      self.assertEqual(b_exit_op_list, [
          "gradients/dense/BiasAdd_grad/tuple/control_dependency_1",
          "gradients/dense/MatMul_grad/tuple/control_dependency_1"
      ])

      # check second taskgraph.
      self.assertFalse(g.taskgraphs[1].is_first_stage)
      self.assertTrue(g.pipeline_enabled)
      self.assertEqual(g.taskgraphs[1].pipeline_config.num_micro_batch, 2)
      self.assertEqual(g.taskgraphs[1].num_replicas, 2)
      self.assertEqual(g.taskgraphs[1].local_num_replicas, 2)
      broadcast_tensors_0 = \
          [tensor.name for tensor in g.taskgraphs[1].get_variables(0)]
      self.assertEqual(broadcast_tensors_0, [
          "dense_1/kernel:0", "dense_1/bias:0", "dense_1/kernel/Adam:0",
          "dense_1/kernel/Adam_1:0", "dense_1/bias/Adam:0",
          "dense_1/bias/Adam_1:0"
      ])
      broadcast_tensors_1 = \
          [tensor.name for tensor in g.taskgraphs[1].get_variables(1)]
      self.assertEqual(broadcast_tensors_1, [
          "EPL_REPLICA_1/dense_1/kernel:0", "EPL_REPLICA_1/dense_1/bias:0",
          "EPL_REPLICA_1/dense_1/kernel/Adam:0",
          "EPL_REPLICA_1/dense_1/kernel/Adam_1:0",
          "EPL_REPLICA_1/dense_1/bias/Adam:0",
          "EPL_REPLICA_1/dense_1/bias/Adam_1:0"
      ])

      gradients = [grad.name for grad in g.taskgraphs[1].gradients]
      list.sort(gradients)
      self.assertEqual(gradients, [
          "gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1:0",
          "gradients/dense_1/MatMul_grad/tuple/control_dependency_1:0"
      ])
      f_ent_ops = [f_ent_op.name for f_ent_op in \
                  list(g.taskgraphs[1].forward_entrance_ops(0, 0))]
      list.sort(f_ent_ops)
      self.assertEqual(len(f_ent_ops), 1)
      self.assertEqual(f_ent_ops, ["dense_1/MatMul"])
      f_exit_ops = [f_exit_op.name for f_exit_op in \
                    list(g.taskgraphs[1].forward_exit_ops(0, 0))]
      list.sort(f_exit_ops)
      self.assertEqual(len(f_exit_ops), 1)
      self.assertListEqual(f_exit_ops, ["Mean"])
      b_ent_ops = [b_ent_op.name for b_ent_op in \
                   list(g.taskgraphs[1].backward_entrance_ops(0, 0))]
      list.sort(b_ent_ops)
      self.assertEqual(len(b_ent_ops), 1)
      self.assertEqual(b_ent_ops, ["gradients/dense_1/MatMul_grad/MatMul"])

      self.assertEqual(len(g.taskgraphs[1].backward_exit_ops(0, 0)), 2)
      s_1_b_exit_0_0 = \
          [b_exit_op.name for b_exit_op in \
           list(g.taskgraphs[1].backward_exit_ops(0, 0))]
      list.sort(s_1_b_exit_0_0)
      self.assertEqual(s_1_b_exit_0_0, [
          "gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1",
          "gradients/dense_1/MatMul_grad/tuple/control_dependency_1"
      ])

      # check forward_operation_placement
      cpu_device = Env.get().cluster.current_worker_cpu()
      for sub_idx in range(len(g.taskgraphs)):
        for op in g.taskgraphs[sub_idx].operations.forward_operations(0, 0):
          if op in dataset_depend_ops:
            self.assertEqual(op.device, cpu_device)
          else:
            self.assertEqual(op.device, g.taskgraphs[sub_idx].virtual_device.get_device(0, 0))

      # check fetch_micro_batch_num
      self.assertEqual(g.get_pipeline_config().num_micro_batch, 2)

      # check num_constructors
      self.assertEqual(g.num_constructors, 1)

      # check if some op need clone
      self.assertEqual(
          g.vars_related_op_names,
          ["dense/kernel", "dense/bias", "dense_1/kernel", "dense_1/bias"])

      # check graphkeys
      self.assertEqual(GraphKeys.ALL_COLLECTION_KEYS, [
          GraphKeys.GLOBAL_CONCAT_OBJECTS, GraphKeys.GLOBAL_MEAN_OBJECTS,
          GraphKeys.GLOBAL_SUM_OBJECTS, GraphKeys.LOCAL_CONCAT_OBJECTS,
          GraphKeys.LOCAL_MEAN_OBJECTS, GraphKeys.LOCAL_SUM_OBJECTS
      ])

      # check collection
      self.assertEqual(g.get_collection(GraphKeys.GLOBAL_CONCAT_OBJECTS), [])
      global_mean = [
          obj.name for obj in g.get_collection(GraphKeys.GLOBAL_MEAN_OBJECTS)
      ]
      self.assertEqual(global_mean, ["Mean:0"])
      self.assertEqual(g.get_collection(GraphKeys.GLOBAL_SUM_OBJECTS), [])
      self.assertEqual(g.get_collection(GraphKeys.LOCAL_CONCAT_OBJECTS), [])
      self.assertEqual(g.get_collection(GraphKeys.LOCAL_MEAN_OBJECTS), [])
      self.assertEqual(g.get_collection(GraphKeys.LOCAL_SUM_OBJECTS), [])

      # check node_clone_for_pipeline
      num_micro_batch = 2
      dp_index = 0
      for sub_idx in range(len(g.taskgraphs) - 1):
        op_list = g.taskgraphs[sub_idx].operations.forward_operations(0, 0)
        for op in op_list:
          for micro_batch_idx in range(1, num_micro_batch):
            prefix = constant.MICRO_BATCH_PREFIX_FORMAT.format(micro_batch_idx)
            node_def = copy.deepcopy(op.node_def)
            if op in dataset_depend_ops:
              continue
            if dp_index == 0 and not g.is_global_step_related(op) \
                and not g.is_vars_related(op):
              op_cloned = g.get_operation_by_name(prefix + op.name)
              self.assertEqual(op_cloned.device, op.device)

      # check dep_ops of pipeline
      customs = []
      for taskgraph in g.taskgraphs:
        customs.append(Custom(taskgraph, 0))

      self.assertEqual([op.name for op in customs[0].forward_entrance_ops[0]],
                       ["IteratorGetNext"])
      f_exit_ops = [op.name for op in customs[0].forward_exit_ops[0]]
      list.sort(f_exit_ops)
      self.assertEqual(f_exit_ops, ["dense/BiasAdd"])
      c_0_b_ent_ops = [op.name for op in customs[0].backward_entrance_ops[0]]
      list.sort(c_0_b_ent_ops)
      self.assertEqual(c_0_b_ent_ops, [
          "gradients/dense/MatMul_grad/MatMul",
          "gradients/dense/MatMul_grad/MatMul_1"
      ])
      c_0_b_exit_ops = [op.name for op in customs[0].backward_exit_ops[0]]
      list.sort(c_0_b_exit_ops)
      self.assertEqual(c_0_b_exit_ops, [
          "gradients/dense/BiasAdd_grad/tuple/control_dependency_1",
          "gradients/dense/MatMul_grad/tuple/control_dependency_1"
      ])
      f_ent_ops = [op.name for op in customs[1].forward_entrance_ops[0]]
      list.sort(f_ent_ops)
      f_exit_ops = [op.name for op in customs[1].forward_exit_ops[0]]
      list.sort(f_exit_ops)
      b_ent_ops = [op.name for op in customs[1].backward_entrance_ops[0]]
      list.sort(b_ent_ops)
      self.assertEqual(f_ent_ops, ["dense_1/MatMul"])
      self.assertEqual(f_exit_ops, ["Mean"])
      self.assertEqual(b_ent_ops, ["gradients/dense_1/MatMul_grad/MatMul"])
      c_1_b_exit_ops_0 = [op.name for op in customs[1].backward_exit_ops[0]]
      list.sort(c_1_b_exit_ops_0)
      self.assertEqual(c_1_b_exit_ops_0, [
          "gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1",
          "gradients/dense_1/MatMul_grad/tuple/control_dependency_1"
      ])
      f_ent_ops = [op.name for op in customs[0].forward_entrance_ops[1]]
      list.sort(f_ent_ops)
      f_exit_ops = [op.name for op in customs[0].forward_exit_ops[1]]
      list.sort(f_exit_ops)
      c_1_b_ent_ops = [op.name for op in customs[0].backward_entrance_ops[1]]
      list.sort(c_1_b_ent_ops)
      c_1_b_exit_ops = [op.name for op in customs[0].backward_exit_ops[1]]
      list.sort(c_1_b_exit_ops)
      self.assertEqual(f_ent_ops, ["EPL_MICRO_BATCH_1/IteratorGetNext"])
      self.assertEqual(f_exit_ops, ["EPL_MICRO_BATCH_1/dense/BiasAdd"])
      self.assertEqual(c_1_b_ent_ops, [
          "EPL_MICRO_BATCH_1/gradients/dense/MatMul_grad/MatMul",
          "EPL_MICRO_BATCH_1/gradients/dense/MatMul_grad/MatMul_1"
      ])
      self.assertEqual(
          c_1_b_exit_ops,
          ["EPL_MICRO_BATCH_1/gradients/dense/BiasAdd_grad/tuple/control_dependency_1", \
           "EPL_MICRO_BATCH_1/gradients/dense/MatMul_grad/tuple/control_dependency_1"])
      f_ent_ops = [op.name for op in customs[1].forward_entrance_ops[1]]
      list.sort(f_ent_ops)
      f_exit_ops = [op.name for op in customs[1].forward_exit_ops[1]]
      list.sort(f_exit_ops)
      c_1_b_ent_ops = [op.name for op in customs[1].backward_entrance_ops[1]]
      list.sort(c_1_b_ent_ops)
      c_1_b_exit_ops = [op.name for op in customs[1].backward_exit_ops[1]]
      list.sort(c_1_b_exit_ops)
      self.assertEqual(f_ent_ops, ["EPL_MICRO_BATCH_1/dense_1/MatMul"])
      self.assertEqual(f_exit_ops, ["EPL_MICRO_BATCH_1/Mean"])
      self.assertEqual(
          c_1_b_ent_ops,
          ["EPL_MICRO_BATCH_1/gradients/dense_1/MatMul_grad/MatMul"])
      self.assertEqual(c_1_b_exit_ops, [
          "EPL_MICRO_BATCH_1/gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1",
          "EPL_MICRO_BATCH_1/gradients/dense_1/MatMul_grad/tuple/control_dependency_1"
      ])

  def test_graph_format(self):
    config = epl.Config()
    config.pipeline.num_micro_batch = 2
    epl.init(config)
    with tf.Graph().as_default():
      loss = self._model_def()
      epl.add_to_collection(loss, GraphKeys.GLOBAL_MEAN_OBJECTS)
      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      gvs = optimizer.compute_gradients(loss)
      optimizer.apply_gradients(gvs)
      is_version_valid = (Version(__version__) >= Version("1.12.0") and Version(__version__) < Version("1.14.0")) or \
          (Version(__version__) < Version("2.0") and Version(__version__) >= Version("1.14.0"))
      self.assertTrue(is_version_valid)
      new_tf_version = Version(__version__) < Version("2.0") and Version(
          __version__) >= Version("1.14.0")

      format_str = "======= Begin Taskgraph 0 replica 0 [Device: /job:worker/replica:0/task:0/device:GPU:0] =======\n" \
                 + "  TensorSliceDataset\n" \
                 + "  BatchDatasetV2\n" \
                 + "  RepeatDataset\n" \
                 + ("  OptimizeDataset\n" if new_tf_version else "") \
                 + ("  ModelDataset\n" if new_tf_version else "") \
                 + "  MakeIterator\n" \
                 + "  IteratorToStringHandle\n" \
                 + "  IteratorGetNext\n" \
                 + "  dense\n" \
                 + "======= End Taskgraph 0 replica 0 [Device: /job:worker/replica:0/task:0/device:GPU:0] =======\n" \
                 + "\n" \
                 + "======= Begin Taskgraph 0 replica 1 [Device: /job:worker/replica:0/task:0/device:GPU:2] =======\n" \
                 + "  EPL_REPLICA_1\n" \
                 + "    EPL_REPLICA_1/IteratorGetNext\n" \
                 + "    EPL_REPLICA_1/dense\n" \
                 + "======= End Taskgraph 0 replica 1 [Device: /job:worker/replica:0/task:0/device:GPU:2] =======\n" \
                 + "\n" \
                 + "======= Begin Taskgraph 1 replica 0 [Device: /job:worker/replica:0/task:0/device:GPU:1] =======\n" \
                 + "  dense_1\n" \
                 + "  Mean\n" \
                 + "======= End Taskgraph 1 replica 0 [Device: /job:worker/replica:0/task:0/device:GPU:1] =======\n" \
                 + "\n" \
                 + "======= Begin Taskgraph 1 replica 1 [Device: /job:worker/replica:0/task:0/device:GPU:3] =======\n" \
                 + "  EPL_REPLICA_1\n" \
                 + "    EPL_REPLICA_1/dense_1\n" \
                 + "    EPL_REPLICA_1/Mean\n" \
                 + "======= End Taskgraph 1 replica 1 [Device: /job:worker/replica:0/task:0/device:GPU:3] ======="

      format_str2 = "======= Begin Taskgraph 0 replica 0 [Device: /job:worker/replica:0/task:0/device:GPU:0] =======\n" \
                  + "  dense\n" \
                  + "======= End Taskgraph 0 replica 0 [Device: /job:worker/replica:0/task:0/device:GPU:0] =======\n" \
                  + "\n" \
                  + "======= Begin Taskgraph 0 replica 1 [Device: /job:worker/replica:0/task:0/device:GPU:2] =======\n" \
                  + "  EPL_REPLICA_1\n" \
                  + "    EPL_REPLICA_1/dense\n" \
                  + "======= End Taskgraph 0 replica 1 [Device: /job:worker/replica:0/task:0/device:GPU:2] =======\n" \
                  + "\n" \
                  + "======= Begin Taskgraph 1 replica 0 [Device: /job:worker/replica:0/task:0/device:GPU:1] =======\n" \
                  + "  dense_1\n" \
                  + "======= End Taskgraph 1 replica 0 [Device: /job:worker/replica:0/task:0/device:GPU:1] =======\n" \
                  + "\n" \
                  + "======= Begin Taskgraph 1 replica 1 [Device: /job:worker/replica:0/task:0/device:GPU:3] =======\n" \
                  + "  EPL_REPLICA_1\n" \
                  + "    EPL_REPLICA_1/dense_1\n" \
                  + "======= End Taskgraph 1 replica 1 [Device: /job:worker/replica:0/task:0/device:GPU:3] ======="
      with tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
          log_device_placement=False)) as sess:
        assert Graph.get().format(max_depth=1).strip() == format_str.strip()
        assert Graph.get().format(
            max_depth=1, prefix_list=["dense"]).strip() == format_str2.strip()

  def _model(self):
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    max_steps = 3
    hooks = [tf.train.StopAtStepHook(last_step=max_steps)]
    return [loss, train_op, global_step], hooks

  def test_multiple_graph(self):
    epl.init()
    steps = []
    with tf.Graph().as_default():
      with epl.replicate(device_count=1):
        train_opts, hooks = self._model()
      with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        graph = epl.Graph.get()
        self.assertEqual(graph.num_stages, 1)
        self.assertEqual(graph.taskgraphs[0].num_replicas, 4)
        for i in range(3):
          train_loss, _, step = sess.run(train_opts)
          steps.append(step)
    self.assertEqual(steps, [0, 1, 2])
    with tf.Graph().as_default():
      train_opts, hooks = self._model()
      with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        graph = epl.Graph.get()
        self.assertEqual(graph.num_stages, 1)
        self.assertEqual(graph.taskgraphs[0].num_replicas, 4)
        steps = []
        for i in range(3):
          train_loss, step = sess.run([train_opts[0], train_opts[2]])
          steps.append(step)
        self.assertEqual(steps, [0, 0, 0])

  def test_multiple_graph_broadcast(self):
    epl.init()
    steps = []
    with tf.Graph().as_default():
      with epl.replicate(device_count=1):
        train_opts, hooks = self._model()
      with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        graph = epl.Graph.get()
        self.assertEqual(graph.num_stages, 1)
        self.assertEqual(graph.taskgraphs[0].num_replicas, 4)
        for i in range(3):
          train_loss, _, step = sess.run(train_opts)
          steps.append(step)
      self.assertEqual(steps, [0, 1, 2])
      broadcast_ops = [o for o in epl.ir.graph.Graph.get().operations.values() if o.type == 'EplNcclCommunicatorBroadcast']
      self.assertEqual(len(broadcast_ops), len(epl.Graph.get().gradients))
    steps = []
    with tf.Graph().as_default():
      with epl.replicate(device_count=1):
        train_opts, hooks = self._model()
      with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        graph = epl.Graph.get()
        self.assertEqual(graph.num_stages, 1)
        self.assertEqual(graph.taskgraphs[0].num_replicas, 4)
        for i in range(3):
          train_loss, _, step = sess.run(train_opts)
          steps.append(step)
      self.assertEqual(steps, [0, 1, 2])
      broadcast_ops = [o for o in epl.ir.graph.Graph.get().operations.values() if o.type == 'EplNcclCommunicatorBroadcast']
      self.assertEqual(len(broadcast_ops), len(epl.Graph.get().gradients))

  def test_dp_define_op_without_taskgraph(self):
    steps = []
    epl.init()
    with tf.Graph().as_default():
      global_step = tf.train.get_or_create_global_step()
      with epl.replicate(device_count=1):
        train_opts, hooks = self._model()
      with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        graph = epl.Graph.get()
        self.assertEqual(graph.num_stages, 1)
        self.assertEqual(graph.taskgraphs[0].num_replicas, 4)
        self.assertEqual(graph.operations[global_step.op.name].taskgraph.index, 0)
        for i in range(3):
          train_loss, _, step = sess.run(train_opts)
          steps.append(step)
      self.assertEqual(steps, [0, 1, 2])

  def test_ga_define_op_without_taskgraph(self):
    steps = []
    config = epl.Config()
    config.pipeline.num_micro_batch = 2
    epl.init(config)
    with tf.Graph().as_default():
      global_step = tf.train.get_or_create_global_step()
      with epl.replicate(device_count=1):
        train_opts, hooks = self._model()
      with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
        graph = epl.Graph.get()
        self.assertEqual(graph.num_stages, 1)
        self.assertEqual(graph.taskgraphs[0].num_replicas, 4)
        self.assertEqual(graph.operations[global_step.op.name].taskgraph.index, 0)
        for i in range(6):
          train_loss, _, step = sess.run(train_opts)
          steps.append(step)
      self.assertEqual(steps, [0, 0, 1, 1, 2, 2])

  def test_multithread_prefetch(self):
    if Version(__version__) >= Version("1.15.0"):
      return
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(2).repeat(1)
    dataset = threadpool.override_threadpool(
          dataset,
          threadpool.PrivateThreadPool(2, display_name='input_pipeline_thread_pool'))
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    source, target = iterator.get_next()
    x = tf.layers.dense(inputs=source, units=16, activation=None)
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    steps = []
    with tf.train.MonitoredTrainingSession() as sess:
      for i in range(3):
        train_loss, _, step = sess.run([loss, train_op, global_step])
        steps.append(step)
    self.assertEqual(steps, [0, 1, 2])

  def test_check_and_set_cloned_dataset_need_clone(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(2).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    source, target = iterator.get_next()
    x = tf.layers.dense(inputs=source, units=16, activation=None)
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    epl.Graph.get().check_and_set_cloned_dataset_need_clone()



# pylint: enable=missing-docstring,protected-access,unused-argument,
# pylint: enable=line-too-long,bad-continuation,unused-variable

if __name__ == "__main__":
  test.main()
