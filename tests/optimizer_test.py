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
"""Test for operations in taskgraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from distutils.version import LooseVersion as Version
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework.versions import __version__

import epl
from epl.cluster import Cluster
from epl.ir.graph import Graph
from epl.ir.phase import ModelPhase
from epl.utils.common import get_device_string

# pylint: disable=missing-docstring,protected-access,unused-argument,line-too-long,bad-continuation,abstract-method
_GPU_PER_WORKER = 8


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))
      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


def _mock_available_gpus():
  def available_gpus(self, *args, **kwargs):
    devices = []
    for gpu_index in range(_GPU_PER_WORKER):
      devices.append(get_device_string(task=0, device_index=gpu_index))
    return devices

  return available_gpus


Cluster.available_gpus = _mock_available_gpus()


class OptimizerTest(test.TestCase):
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
      x, y = iterator.get_next()
      dense1 = tf.layers.dense(inputs=x, units=10, activation=None)
      emb = tf.get_variable("emb", [100, 16], trainable=True)
      lookup_values = tf.nn.embedding_lookup(emb, y)
      mat_res = tf.matmul(dense1, lookup_values)
    with epl.replicate(device_count=1, name="stage_1"):
      logits = tf.layers.dense(inputs=mat_res, units=10, activation=None)
      return tf.reduce_mean(logits)

  def test_adam(self):
    config = epl.Config()
    config.pipeline.num_micro_batch = 3
    epl.init(config)
    with tf.Graph().as_default():
      loss = self._model_def()
      graph = Graph.get()
      self.assertEqual(graph._current_model_phase, ModelPhase.FORWARD)

      optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
      gvs = optimizer.compute_gradients(loss)
      optimizer.apply_gradients(gvs)

      tf.train.MonitoredTrainingSession(config=tf.ConfigProto(
          log_device_placement=False))

      # check first taskgraph.
      forward_operations = [
          op.name
          for op in graph.taskgraphs[0].operations.forward_operations(0, 0)
      ]
      list.sort(forward_operations)
      if Version(__version__) >= Version("1.12.0") and \
          Version(__version__) < Version("1.14.0"):
        self.assertListEqual(forward_operations, [
            "BatchDatasetV2", "IteratorGetNext", "IteratorToStringHandle",
            "IteratorV2", "MakeIterator", "MatMul", "RepeatDataset",
            "TensorSliceDataset", "batch_size", "count", "dense/BiasAdd",
            "dense/MatMul", "dense/bias", "dense/bias/Assign",
            "dense/bias/Initializer/zeros", "dense/bias/read", "dense/kernel",
            "dense/kernel/Assign", "dense/kernel/Initializer/random_uniform",
            "dense/kernel/Initializer/random_uniform/RandomUniform",
            "dense/kernel/Initializer/random_uniform/max",
            "dense/kernel/Initializer/random_uniform/min",
            "dense/kernel/Initializer/random_uniform/mul",
            "dense/kernel/Initializer/random_uniform/shape",
            "dense/kernel/Initializer/random_uniform/sub", "dense/kernel/read",
            "drop_remainder", "emb", "emb/Assign",
            "emb/Initializer/random_uniform",
            "emb/Initializer/random_uniform/RandomUniform",
            "emb/Initializer/random_uniform/max",
            "emb/Initializer/random_uniform/min",
            "emb/Initializer/random_uniform/mul",
            "emb/Initializer/random_uniform/shape",
            "emb/Initializer/random_uniform/sub", "emb/read",
            "embedding_lookup", "embedding_lookup/Identity",
            "embedding_lookup/axis", "tensors/component_0",
            "tensors/component_1"
        ])

      elif Version(__version__) < Version("2.0"):
        self.assertListEqual(forward_operations, [
            "BatchDatasetV2", "IteratorGetNext", "IteratorToStringHandle",
            "IteratorV2", "MakeIterator", "MatMul", "ModelDataset",
            "OptimizeDataset", "RepeatDataset", "TensorSliceDataset",
            "batch_size", "count", "dense/BiasAdd", "dense/MatMul",
            "dense/bias", "dense/bias/Assign", "dense/bias/Initializer/zeros",
            "dense/bias/read", "dense/kernel", "dense/kernel/Assign",
            "dense/kernel/Initializer/random_uniform",
            "dense/kernel/Initializer/random_uniform/RandomUniform",
            "dense/kernel/Initializer/random_uniform/max",
            "dense/kernel/Initializer/random_uniform/min",
            "dense/kernel/Initializer/random_uniform/mul",
            "dense/kernel/Initializer/random_uniform/shape",
            "dense/kernel/Initializer/random_uniform/sub", "dense/kernel/read",
            "drop_remainder", "emb", "emb/Assign",
            "emb/Initializer/random_uniform",
            "emb/Initializer/random_uniform/RandomUniform",
            "emb/Initializer/random_uniform/max",
            "emb/Initializer/random_uniform/min",
            "emb/Initializer/random_uniform/mul",
            "emb/Initializer/random_uniform/shape",
            "emb/Initializer/random_uniform/sub", "emb/read",
            "embedding_lookup", "embedding_lookup/Identity",
            "embedding_lookup/axis", "normalize_element/component_0",
            "normalize_element/component_1", "optimizations"
        ])
      else:
        raise RuntimeError("Version of tensorflow is not supported for now."
                           "Tenosrflow Version: %s." % __version__)
      backward_operations = [
          op.name
          for op in graph.taskgraphs[0].operations.backward_operations(0, 0)
      ]
      list.sort(backward_operations)
      if Version(__version__) >= Version("1.12.0") and \
          Version(__version__) < Version("1.14.0"):
        self.assertEqual(backward_operations, [
            "gradients/MatMul_grad/MatMul", "gradients/MatMul_grad/MatMul_1",
            "gradients/MatMul_grad/tuple/control_dependency",
            "gradients/MatMul_grad/tuple/control_dependency_1",
            "gradients/MatMul_grad/tuple/group_deps",
            "gradients/dense/BiasAdd_grad/BiasAddGrad",
            "gradients/dense/BiasAdd_grad/tuple/control_dependency",
            "gradients/dense/BiasAdd_grad/tuple/control_dependency_1",
            "gradients/dense/BiasAdd_grad/tuple/group_deps",
            "gradients/dense/MatMul_grad/MatMul",
            "gradients/dense/MatMul_grad/MatMul_1",
            "gradients/dense/MatMul_grad/tuple/control_dependency",
            "gradients/dense/MatMul_grad/tuple/control_dependency_1",
            "gradients/dense/MatMul_grad/tuple/group_deps",
            "gradients/embedding_lookup_grad/ExpandDims",
            "gradients/embedding_lookup_grad/ExpandDims/dim",
            "gradients/embedding_lookup_grad/Reshape",
            "gradients/embedding_lookup_grad/Reshape_1",
            "gradients/embedding_lookup_grad/Shape",
            "gradients/embedding_lookup_grad/Size",
            "gradients/embedding_lookup_grad/ToInt32",
            "gradients/embedding_lookup_grad/concat",
            "gradients/embedding_lookup_grad/concat/axis",
            "gradients/embedding_lookup_grad/strided_slice",
            "gradients/embedding_lookup_grad/strided_slice/stack",
            "gradients/embedding_lookup_grad/strided_slice/stack_1",
            "gradients/embedding_lookup_grad/strided_slice/stack_2",
            "gradients_1/UnsortedSegmentSum", "gradients_1/strided_slice",
            "gradients_1/strided_slice/stack",
            "gradients_1/strided_slice/stack_1",
            "gradients_1/strided_slice/stack_2"
        ])
      elif Version(__version__) < Version("2.0"):
        self.assertEqual(backward_operations, [
            "gradients/MatMul_grad/MatMul", "gradients/MatMul_grad/MatMul_1",
            "gradients/MatMul_grad/tuple/control_dependency",
            "gradients/MatMul_grad/tuple/control_dependency_1",
            "gradients/MatMul_grad/tuple/group_deps",
            "gradients/dense/BiasAdd_grad/BiasAddGrad",
            "gradients/dense/BiasAdd_grad/tuple/control_dependency",
            "gradients/dense/BiasAdd_grad/tuple/control_dependency_1",
            "gradients/dense/BiasAdd_grad/tuple/group_deps",
            "gradients/dense/MatMul_grad/MatMul",
            "gradients/dense/MatMul_grad/MatMul_1",
            "gradients/dense/MatMul_grad/tuple/control_dependency",
            "gradients/dense/MatMul_grad/tuple/control_dependency_1",
            "gradients/dense/MatMul_grad/tuple/group_deps",
            "gradients/embedding_lookup_grad/Cast",
            "gradients/embedding_lookup_grad/ExpandDims",
            "gradients/embedding_lookup_grad/ExpandDims/dim",
            "gradients/embedding_lookup_grad/Reshape",
            "gradients/embedding_lookup_grad/Reshape_1",
            "gradients/embedding_lookup_grad/Shape",
            "gradients/embedding_lookup_grad/Size",
            "gradients/embedding_lookup_grad/concat",
            "gradients/embedding_lookup_grad/concat/axis",
            "gradients/embedding_lookup_grad/strided_slice",
            "gradients/embedding_lookup_grad/strided_slice/stack",
            "gradients/embedding_lookup_grad/strided_slice/stack_1",
            "gradients/embedding_lookup_grad/strided_slice/stack_2",
            "gradients_1/UnsortedSegmentSum", "gradients_1/strided_slice",
            "gradients_1/strided_slice/stack",
            "gradients_1/strided_slice/stack_1",
            "gradients_1/strided_slice/stack_2"
        ])
      else:
        raise RuntimeError("Version of tensorflow is not supported for now."
                           "Tenosrflow Version: %s." % __version__)

      apply_operations = [
          op.name for op in graph.taskgraphs[0].operations.apply_operations(0)
      ]
      list.sort(apply_operations)
      if Version(__version__) >= Version("1.12.0") and \
          Version(__version__) < Version("1.14.0"):
        self.assertListEqual(apply_operations, [
            "Adam/beta1", "Adam/beta2", "Adam/epsilon", "Adam/learning_rate",
            "Adam/update_dense/bias/ApplyAdam",
            "Adam/update_dense/kernel/ApplyAdam", "Adam/update_emb/ApplyAdam",
            "beta1_power", "beta1_power/Assign", "beta1_power/initial_value",
            "beta1_power/read", "beta2_power", "beta2_power/Assign",
            "beta2_power/initial_value", "beta2_power/read", "dense/bias/Adam",
            "dense/bias/Adam/Assign", "dense/bias/Adam/Initializer/zeros",
            "dense/bias/Adam/read", "dense/bias/Adam_1",
            "dense/bias/Adam_1/Assign", "dense/bias/Adam_1/Initializer/zeros",
            "dense/bias/Adam_1/read", "dense/kernel/Adam",
            "dense/kernel/Adam/Assign", "dense/kernel/Adam/Initializer/zeros",
            "dense/kernel/Adam/read", "dense/kernel/Adam_1",
            "dense/kernel/Adam_1/Assign",
            "dense/kernel/Adam_1/Initializer/zeros",
            "dense/kernel/Adam_1/read", "emb/Adam", "emb/Adam/Assign",
            "emb/Adam/Initializer/zeros", "emb/Adam/Initializer/zeros/Const",
            "emb/Adam/Initializer/zeros/shape_as_tensor", "emb/Adam/read",
            "emb/Adam_1", "emb/Adam_1/Assign", "emb/Adam_1/Initializer/zeros",
            "emb/Adam_1/Initializer/zeros/Const",
            "emb/Adam_1/Initializer/zeros/shape_as_tensor", "emb/Adam_1/read"
        ])
      elif Version(__version__) < Version("2.0"):
        self.assertListEqual(apply_operations, [
            "Adam/beta1", "Adam/beta2", "Adam/epsilon", "Adam/learning_rate",
            "Adam/update_dense/bias/ApplyAdam",
            "Adam/update_dense/kernel/ApplyAdam", "Adam/update_emb/ApplyAdam",
            "beta1_power", "beta1_power/Assign", "beta1_power/initial_value",
            "beta1_power/read", "beta2_power", "beta2_power/Assign",
            "beta2_power/initial_value", "beta2_power/read", "dense/bias/Adam",
            "dense/bias/Adam/Assign", "dense/bias/Adam/Initializer/zeros",
            "dense/bias/Adam/read", "dense/bias/Adam_1",
            "dense/bias/Adam_1/Assign", "dense/bias/Adam_1/Initializer/zeros",
            "dense/bias/Adam_1/read", "dense/kernel/Adam",
            "dense/kernel/Adam/Assign", "dense/kernel/Adam/Initializer/zeros",
            "dense/kernel/Adam/read", "dense/kernel/Adam_1",
            "dense/kernel/Adam_1/Assign",
            "dense/kernel/Adam_1/Initializer/zeros",
            "dense/kernel/Adam_1/read", "emb/Adam", "emb/Adam/Assign",
            "emb/Adam/Initializer/zeros", "emb/Adam/Initializer/zeros/Const",
            "emb/Adam/Initializer/zeros/shape_as_tensor", "emb/Adam/read",
            "emb/Adam_1", "emb/Adam_1/Assign", "emb/Adam_1/Initializer/zeros",
            "emb/Adam_1/Initializer/zeros/Const",
            "emb/Adam_1/Initializer/zeros/shape_as_tensor", "emb/Adam_1/read"
        ])
      else:
        raise RuntimeError("Version of tensorflow is not supported for now."
                           "Tenosrflow Version: %s." % __version__)

      # check second taskgraph.
      forward_operations = [
          op.name
          for op in graph.taskgraphs[1].operations.forward_operations(0, 0)
      ]
      list.sort(forward_operations)
      self.assertEqual(forward_operations,
          ["Const", "Mean", "dense_1/BiasAdd", "dense_1/MatMul", \
          "dense_1/bias", "dense_1/bias/Assign", \
          "dense_1/bias/Initializer/zeros", "dense_1/bias/read", \
          "dense_1/kernel", "dense_1/kernel/Assign", \
          "dense_1/kernel/Initializer/random_uniform", \
          "dense_1/kernel/Initializer/random_uniform/RandomUniform", \
          "dense_1/kernel/Initializer/random_uniform/max", \
          "dense_1/kernel/Initializer/random_uniform/min", \
          "dense_1/kernel/Initializer/random_uniform/mul", \
          "dense_1/kernel/Initializer/random_uniform/shape", \
          "dense_1/kernel/Initializer/random_uniform/sub", \
          "dense_1/kernel/read"])
      backward_operations = [
          op.name
          for op in graph.taskgraphs[1].operations.backward_operations(0, 0)
      ]
      list.sort(backward_operations)
      self.assertEqual(backward_operations,
          ["gradients/Fill", "gradients/Mean_grad/Cast", \
          "gradients/Mean_grad/Const", "gradients/Mean_grad/Const_1", \
          "gradients/Mean_grad/Maximum", "gradients/Mean_grad/Maximum/y", \
          "gradients/Mean_grad/Prod", "gradients/Mean_grad/Prod_1", \
          "gradients/Mean_grad/Reshape", "gradients/Mean_grad/Reshape/shape", \
          "gradients/Mean_grad/Shape", "gradients/Mean_grad/Shape_1", \
          "gradients/Mean_grad/Shape_2", "gradients/Mean_grad/Tile", \
          "gradients/Mean_grad/floordiv", "gradients/Mean_grad/truediv", \
          "gradients/Shape", "gradients/dense_1/BiasAdd_grad/BiasAddGrad", \
          "gradients/dense_1/BiasAdd_grad/tuple/control_dependency", \
          "gradients/dense_1/BiasAdd_grad/tuple/control_dependency_1", \
          "gradients/dense_1/BiasAdd_grad/tuple/group_deps", \
          "gradients/dense_1/MatMul_grad/MatMul", \
          "gradients/dense_1/MatMul_grad/MatMul_1", \
          "gradients/dense_1/MatMul_grad/tuple/control_dependency", \
          "gradients/dense_1/MatMul_grad/tuple/control_dependency_1", \
          "gradients/dense_1/MatMul_grad/tuple/group_deps", \
          "gradients/grad_ys_0"])
      apply_operations = [
          op.name for op in graph.taskgraphs[1].operations.apply_operations(0)
      ]
      list.sort(apply_operations)
      if Version(__version__) >= Version("1.12.0") and \
          Version(__version__) < Version("1.14.0"):
        self.assertEqual(apply_operations, [
            "Adam", "Adam/Assign", "Adam/Assign_1", "Adam/NoOp", "Adam/NoOp_1",
            "Adam/mul", "Adam/mul_1", "Adam/update_dense_1/bias/ApplyAdam",
            "Adam/update_dense_1/kernel/ApplyAdam", "dense_1/bias/Adam",
            "dense_1/bias/Adam/Assign", "dense_1/bias/Adam/Initializer/zeros",
            "dense_1/bias/Adam/read", "dense_1/bias/Adam_1",
            "dense_1/bias/Adam_1/Assign",
            "dense_1/bias/Adam_1/Initializer/zeros",
            "dense_1/bias/Adam_1/read", "dense_1/kernel/Adam",
            "dense_1/kernel/Adam/Assign",
            "dense_1/kernel/Adam/Initializer/zeros",
            "dense_1/kernel/Adam/read", "dense_1/kernel/Adam_1",
            "dense_1/kernel/Adam_1/Assign",
            "dense_1/kernel/Adam_1/Initializer/zeros",
            "dense_1/kernel/Adam_1/read"
        ])
      elif Version(__version__) < Version("2.0"):
        # Test only for nvidia-tf 1.15.4.
        if Version(__version__) == Version("1.15.4"):
          self.assertEqual(apply_operations, [
              "Adam/Adam/-apply", "Adam/Adam/-apply/NoOp",
              "Adam/Adam/-apply/NoOp_1", "Adam/Assign", "Adam/Assign_1",
              "Adam/mul", "Adam/mul_1", "Adam/update_dense_1/bias/ApplyAdam",
              "Adam/update_dense_1/kernel/ApplyAdam", "dense_1/bias/Adam",
              "dense_1/bias/Adam/Assign",
              "dense_1/bias/Adam/Initializer/zeros", "dense_1/bias/Adam/read",
              "dense_1/bias/Adam_1", "dense_1/bias/Adam_1/Assign",
              "dense_1/bias/Adam_1/Initializer/zeros",
              "dense_1/bias/Adam_1/read", "dense_1/kernel/Adam",
              "dense_1/kernel/Adam/Assign",
              "dense_1/kernel/Adam/Initializer/zeros",
              "dense_1/kernel/Adam/read", "dense_1/kernel/Adam_1",
              "dense_1/kernel/Adam_1/Assign",
              "dense_1/kernel/Adam_1/Initializer/zeros",
              "dense_1/kernel/Adam_1/read"
          ])
        else:
          self.assertEqual(apply_operations, [
              "Adam", "Adam/Assign", "Adam/Assign_1", "Adam/NoOp",
              "Adam/NoOp_1", "Adam/mul", "Adam/mul_1",
              "Adam/update_dense_1/bias/ApplyAdam",
              "Adam/update_dense_1/kernel/ApplyAdam", "dense_1/bias/Adam",
              "dense_1/bias/Adam/Assign",
              "dense_1/bias/Adam/Initializer/zeros", "dense_1/bias/Adam/read",
              "dense_1/bias/Adam_1", "dense_1/bias/Adam_1/Assign",
              "dense_1/bias/Adam_1/Initializer/zeros",
              "dense_1/bias/Adam_1/read", "dense_1/kernel/Adam",
              "dense_1/kernel/Adam/Assign",
              "dense_1/kernel/Adam/Initializer/zeros",
              "dense_1/kernel/Adam/read", "dense_1/kernel/Adam_1",
              "dense_1/kernel/Adam_1/Assign",
              "dense_1/kernel/Adam_1/Initializer/zeros",
              "dense_1/kernel/Adam_1/read"
          ])

  def test_custom_optimizer(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(-1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = AdamWeightDecayOptimizer(
        learning_rate=0.01,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
    code = optimizer.__class__.apply_gradients.__code__
    self.assertTrue('epl/parallel/hooks.py' in code.co_filename)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    with tf.train.MonitoredTrainingSession() as sess:
      for _ in range(5):
        sess.run([loss, train_op, global_step])
    self.assertEqual(len(epl.Graph.get().gradients), 8)

  def test_custom_optimizer2(self):
    class CustomOptimizer(tf.train.AdamOptimizer):
      def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for (grad, param) in grads_and_vars:
          if grad is None or param is None:
            continue
          update_with_lr = self._lr * grad
          next_param = param - update_with_lr
          assignments.extend([param.assign(next_param)])
        return tf.group(*assignments, name=name)
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(-1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = CustomOptimizer(learning_rate=0.01)
    code = optimizer.__class__.apply_gradients.__code__
    self.assertTrue('epl/parallel/hooks.py' in code.co_filename)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    with tf.train.MonitoredTrainingSession() as sess:
      for _ in range(5):
        sess.run([loss, train_op, global_step])
    self.assertEqual(len(epl.Graph.get().gradients), 8)

  def test_optimizer_post_process(self):
    class CustomOptimizer(tf.train.AdamOptimizer):
      def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        assignments = []
        for (grad, param) in grads_and_vars:
          if grad is None or param is None:
            continue
          update_with_lr = self._lr * grad
          next_param = param - update_with_lr
          assignments.extend([param.assign(next_param)])
        return tf.group(*assignments, name=name)
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(-1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    dense1 = tf.layers.dense(inputs=x, units=16, activation=None)
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None)
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = CustomOptimizer(learning_rate=0.01)
    code = optimizer.__class__.apply_gradients.__code__
    self.assertTrue('epl/parallel/hooks.py' in code.co_filename)
    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    with tf.train.MonitoredTrainingSession() as sess:
      for _ in range(5):
        sess.run([loss, train_op, global_step])
    self.assertEqual(len(epl.Graph.get().gradients), 8)
    self.assertEqual(epl.Graph.get().operations[train_op.name].phase, ModelPhase.APPLY)


# pylint: enable=missing-docstring,protected-access,unused-argument,line-too-long,bad-continuation,abstract-method

if __name__ == "__main__":
  test.main()
