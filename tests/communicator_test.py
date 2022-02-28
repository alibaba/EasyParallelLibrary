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
"""Tests for epl communicators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resources
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging

from epl.cluster import Cluster
from epl.communicators.base import Communicator
from epl.communicators.collective_communicator import CollectiveCommunicator
from epl.env import Env
from epl.parallel.ops import create_serial_communicator
from epl.parallel.ops import create_simple_communicator


# pylint: disable=missing-docstring,unused-variable
class CommunicatorTest(test.TestCase):
  def __init__(self, method_name='runTest'):
    super(CommunicatorTest, self).__init__(method_name)
    self._config = config_pb2.ConfigProto(log_device_placement=False,
                                          allow_soft_placement=True,
                                          gpu_options=config_pb2.GPUOptions(
                                              allow_growth=True,
                                              force_gpu_compatible=True))
    Env.get().cluster = Cluster()

  def test_allreduce_simple_tensor(self):
    a = 13
    b = 22
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices)
        input0 = array_ops.constant(a)
        sum0 = comm0.batch_allreduce(input0)
      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices)
        input1 = array_ops.constant(b)
        sum1 = comm1.batch_allreduce(input1)
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)
      s0, s1 = sess.run([sum0, sum1])
      self.assertAllClose(s0, a + b, rtol=1e-6)
      self.assertAllClose(s1, a + b, rtol=1e-6)

  def test_allreduce_tensors(self):
    a0 = 13
    a1 = 13.1
    b0 = 22
    b1 = 22.2
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices)
        input0_0 = array_ops.constant(a0, dtype=dtypes.int32)
        input0_1 = array_ops.constant(a1, dtype=dtypes.float32)
        sum0 = comm0.batch_allreduce([input0_0, input0_1])
      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices)
        input1_0 = array_ops.constant(b0, dtype=dtypes.int32)
        input1_1 = array_ops.constant(b1, dtype=dtypes.float32)
        sum1 = comm1.batch_allreduce([input1_0, input1_1])
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)
      s0, s1 = sess.run([sum0, sum1])
      self.assertAllClose(s0[0], a0 + b0, rtol=1e-6)
      self.assertAllClose(s0[1], a1 + b1, rtol=1e-6)
      self.assertAllClose(s1[0], a0 + b0, rtol=1e-6)
      self.assertAllClose(s1[1], a1 + b1, rtol=1e-6)

  def test_allreduce_grad(self):
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        a = constant_op.constant(1.0, shape=[2, 10])
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices)
        recv0 = comm0.batch_allreduce(a * 0.75)
        loss0 = math_ops.reduce_mean(recv0) * 20.0
      with ops.device('/gpu:1'):
        b = constant_op.constant(2.0, shape=[2, 10])
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices)
        recv1 = comm1.batch_allreduce(b)
        loss1 = math_ops.reduce_mean(recv1) * 10.0
      loss = loss0 * loss1
      grad0, grad1 = gradients_impl.gradients([loss], [a, b], [2.0],
                                              colocate_gradients_with_ops=True)
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)
      g0, g1 = sess.run([grad0, grad1])
      self.assertAllClose(g0, [[82.5] * 10] * 2, rtol=1e-6)
      self.assertAllClose(g1, [[110.0] * 10] * 2, rtol=1e-6)

  def test_allreduce_tensors_with_mean(self):
    a0 = 13
    a1 = 13.1
    b0 = 22
    b1 = 22.2
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices)
        input0_0 = array_ops.constant(a0, dtype=dtypes.int32)
        input0_1 = array_ops.constant(a1, dtype=dtypes.float32)
        sum0 = comm0.batch_allreduce([input0_0, input0_1], mean=True)
      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices)
        input1_0 = array_ops.constant(b0, dtype=dtypes.int32)
        input1_1 = array_ops.constant(b1, dtype=dtypes.float32)
        sum1 = comm1.batch_allreduce([input1_0, input1_1], mean=True)
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)
      s0, s1 = sess.run([sum0, sum1])
      self.assertEqual(s0[0].dtype, "int32")
      self.assertEqual(s0[1].dtype, "float32")
      self.assertEqual(s1[0].dtype, "int32")
      self.assertEqual(s1[1].dtype, "float32")
      self.assertAllClose(s0[0], (a0 + b0) // 2, rtol=1e-6)
      self.assertAllClose(s0[1], (a1 + b1) / 2, rtol=1e-6)
      self.assertAllClose(s1[0], (a0 + b0) // 2, rtol=1e-6)
      self.assertAllClose(s1[1], (a1 + b1) / 2, rtol=1e-6)

  def broadcast_simple_tensor(self, use_serial_com=False):
    a = 13
    b = 22
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        if use_serial_com:
          comm0 = create_serial_communicator(name=shared_name, devices=devices)
        else:
          comm0 = CollectiveCommunicator(name=shared_name, devices=devices)
        input0 = array_ops.constant(a)
        b0 = comm0.broadcast(input0)
      with ops.device('/gpu:1'):
        if use_serial_com:
          comm1 = create_serial_communicator(name=shared_name, devices=devices)
        else:
          comm1 = CollectiveCommunicator(name=shared_name, devices=devices)
        input1 = array_ops.constant(b)
        b1 = comm1.broadcast(input1)
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)
      r0, r1 = sess.run([b0, b1])
      self.assertAllClose(r0, a, rtol=1e-6)
      self.assertAllClose(r1, a, rtol=1e-6)

  def test_broadcast_simple_tensor(self):
    for use_serial_com in [True, False]:
      self.broadcast_simple_tensor(use_serial_com=use_serial_com)

  def broadcast_tensors(self,
                        dtype1=dtypes.int32,
                        dtype2=dtypes.int32,
                        use_serial_com=False):
    a0 = [11, 12, 13, 14]
    a1 = [[21, 22, 23], [24, 25, 26]]
    b0 = [1, 1, 1, 1]
    b1 = [[1, 1, 1], [1, 1, 1]]
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        if use_serial_com:
          comm0 = create_serial_communicator(name=shared_name, devices=devices)
        else:
          comm0 = CollectiveCommunicator(name=shared_name, devices=devices)
        input0_0 = array_ops.constant(a0, dtype=dtype1)
        input0_1 = array_ops.constant(a1, dtype=dtype2)
        v0 = comm0.broadcast([input0_0, input0_1])
      with ops.device('/gpu:1'):
        if use_serial_com:
          comm1 = create_serial_communicator(name=shared_name, devices=devices)
        else:
          comm1 = CollectiveCommunicator(name=shared_name, devices=devices)
        input1_0 = array_ops.constant(b0, dtype=dtype1)
        input1_1 = array_ops.constant(b1, dtype=dtype2)
        v1 = comm1.broadcast([input1_0, input1_1])
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)
      r0, r1 = sess.run([v0, v1])
      self.assertAllClose(r0[0], a0, rtol=1e-6)
      self.assertAllClose(r0[1], a1, rtol=1e-6)
      self.assertAllClose(r1[0], a0, rtol=1e-6)
      self.assertAllClose(r1[1], a1, rtol=1e-6)

  def test_broadcast_tensors(self):
    self.broadcast_tensors()
    for dtype1 in [dtypes.float32, dtypes.int32]:
      for dtype2 in [dtypes.float32, dtypes.int32]:
        for use_serial_com in [True, False]:
          self.broadcast_tensors(dtype1=dtype1, dtype2=dtype2,
                                 use_serial_com=use_serial_com)

  def test_broadcast_tensors_root1(self):
    a0 = [11, 12, 13, 14]
    a1 = [[21, 22, 23], [24, 25, 26]]
    b0 = [1, 1, 1, 1]
    b1 = [[1, 1, 1], [1, 1, 1]]
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices)
        input0_0 = array_ops.constant(a0)
        input0_1 = array_ops.constant(a1)
        v0 = comm0.broadcast([input0_0, input0_1], root_rank=1)
      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices)
        input1_0 = array_ops.constant(b0)
        input1_1 = array_ops.constant(b1)
        v1 = comm1.broadcast([input1_0, input1_1], root_rank=1)
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)
      r0, r1 = sess.run([v0, v1])
      self.assertAllClose(r0[0], b0, rtol=1e-6)
      self.assertAllClose(r0[1], b1, rtol=1e-6)
      self.assertAllClose(r1[0], b0, rtol=1e-6)
      self.assertAllClose(r1[1], b1, rtol=1e-6)

  def test_broadcast_tensors_reuse(self):
    import os
    os.environ["BRIDGE_ENABLE_TAO"] = "false"
    os.environ["TF_DISABLE_TAO_BRIDGE"] = "true"
    a0 = [11, 12, 13, 14]
    a1 = [[21, 22, 23], [24, 25, 26]]
    b0 = [1, 1, 1, 1]
    b1 = [[1, 1, 1], [1, 1, 1]]
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices, num_communicators=1)
        input0_0 = array_ops.constant(a0)
        input0_1 = array_ops.constant(a1)
        input0_2 = array_ops.constant(a0)
        input0_3 = array_ops.constant(a1)
        v0 = comm0.broadcast([input0_0, input0_1])
        with ops.control_dependencies(v0):
          v01 = comm0.broadcast([input0_2, input0_3])
      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices, num_communicators=1)
        input1_0 = array_ops.constant(b0)
        input1_1 = array_ops.constant(b1)
        input1_2 = array_ops.constant(b0)
        input1_3 = array_ops.constant(b1)
        v1 = comm1.broadcast([input1_0, input1_1])
        with ops.control_dependencies(v1):
          v11 = comm1.broadcast([input1_2, input1_3])
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(
        use_gpu=True, graph=graph, config=self._config) as sess:
      sess.run(init_op)
      r0, r01, r1, r11 = sess.run([v0, v01, v1, v11])
      self.assertAllClose(
          r0[0],
          a0,
          rtol=1e-6)
      self.assertAllClose(
          r0[1],
          a1,
          rtol=1e-6)
      self.assertAllClose(
          r01[0],
          a0,
          rtol=1e-6)
      self.assertAllClose(
          r01[1],
          a1,
          rtol=1e-6)
      self.assertAllClose(
          r1[0],
          a0,
          rtol=1e-6)
      self.assertAllClose(
          r1[1],
          a1,
          rtol=1e-6)
      self.assertAllClose(
          r11[0],
          a0,
          rtol=1e-6)
      self.assertAllClose(
          r11[1],
          a1,
          rtol=1e-6)

  def test_allgather(self):
    a0 = [[1, 2, 3, 4], [9, 10, 11, 12]]
    a1 = [[5, 6, 7, 8], [13, 14, 15, 16]]
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices)
        input0 = array_ops.constant(a0, dtype=dtypes.int32)
        v0 = comm0.allgather(input0)
      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices)
        input1 = array_ops.constant(a1, dtype=dtypes.int32)
        v1 = comm1.allgather(input1)
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)
      r0, r1 = sess.run([v0, v1])
      self.assertAllEqual(r0, a0 + a1)
      self.assertAllEqual(r1, a0 + a1)

  def test_alltoall(self):
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices)
        in1 = array_ops.reshape(math_ops.range(100, 148, delta=1),
                                [2, 3, 2, 4])
        v1 = comm0.alltoall(in1)

      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices)
        in2 = array_ops.reshape(math_ops.range(300, 348, delta=1),
                                [2, 3, 2, 4])
        v2 = comm1.alltoall(in2)

      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)

      r1, r2 = sess.run([v1, v2])
      expected_in1, expected_in2 = sess.run([in1, in2])
      alltoall_input = sess.run([in1, in2])
      expected = [i for i in map(list, zip(*alltoall_input))]

      self.assertAllEqual(r1[0], expected[0][0])
      self.assertAllEqual(r1[1], expected[0][1])
      self.assertAllEqual(r2[0], expected[1][0])
      self.assertAllEqual(r2[1], expected[1][1])

  def test_alltoall_with_reuse_comm(self):
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = create_simple_communicator(shared_name,
                                           devices)
        in1 = array_ops.reshape(math_ops.range(100, 148, delta=1),
                                [2, 3, 2, 4])
        v1 = comm0.alltoall(in1)
        in2 = array_ops.reshape(math_ops.range(200, 248, delta=1),
                                [2, 3, 2, 4])
        with ops.control_dependencies([v1]):
          v2 = comm0.alltoall(in2)

      with ops.device('/gpu:1'):
        comm1 = create_simple_communicator(shared_name,
                                           devices)
        in3 = array_ops.reshape(math_ops.range(300, 348, delta=1),
                                [2, 3, 2, 4])
        v3 = comm1.alltoall(in3)
        in4 = array_ops.reshape(math_ops.range(400, 448, delta=1),
                                [2, 3, 2, 4])
        with ops.control_dependencies([v3]):
          v4 = comm1.alltoall(in4)

      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)

      expected_in1 = sess.run([in1, in3])
      expected_in1 = [i for i in map(list, zip(*expected_in1))]
      expected_in2 = sess.run([in2, in4])
      expected_in2 = [i for i in map(list, zip(*expected_in2))]

      r1, r2, r3, r4 = sess.run([v1, v2, v3, v4])
      self.assertAllEqual(r1[0], expected_in1[0][0])
      self.assertAllEqual(r1[1], expected_in1[0][1])
      self.assertAllEqual(r3[0], expected_in1[1][0])
      self.assertAllEqual(r3[1], expected_in1[1][1])
      self.assertAllEqual(r2[0], expected_in2[0][0])
      self.assertAllEqual(r2[1], expected_in2[0][1])
      self.assertAllEqual(r4[0], expected_in2[1][0])
      self.assertAllEqual(r4[1], expected_in2[1][1])

  def reduce_simple_tensor(self, root_rank=0, reduce_op=Communicator.SUM):
    a0 = [[1, 9, 7, 4], [9, 14, 11, 16]]
    a1 = [[5, 6, 3, 8], [13, 10, 15, 12]]
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices)
        input0 = array_ops.constant(a0, dtype=dtypes.int32)
        v0 = comm0.reduce(input0, root_rank=root_rank, reduce_op=reduce_op)
      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices)
        input1 = array_ops.constant(a1, dtype=dtypes.int32)
        v1 = comm1.reduce(input1, root_rank=root_rank, reduce_op=reduce_op)
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)
      r0, r1 = sess.run([v0, v1])
      if root_rank == 0:
        if reduce_op == Communicator.SUM:
          self.assertAllEqual(r0, np.add(a0, a1))
        elif reduce_op == Communicator.MAX:
          self.assertAllEqual(r0, np.maximum(a0, a1))
      else:
        if reduce_op == Communicator.SUM:
          self.assertAllEqual(r1, np.add(a0, a1))
        elif reduce_op == Communicator.MAX:
          self.assertAllEqual(r1, np.maximum(a0, a1))

  def test_reduce_simple_tensor(self):
    self.reduce_simple_tensor(root_rank=0, reduce_op=Communicator.SUM)
    self.reduce_simple_tensor(root_rank=1, reduce_op=Communicator.SUM)
    self.reduce_simple_tensor(root_rank=0, reduce_op=Communicator.MAX)
    self.reduce_simple_tensor(root_rank=1, reduce_op=Communicator.MAX)

  def reduce_tensors(self, root_rank=0, num_communicator=None):
    a0 = [[1, 9, 7, 4], [9, 14, 11, 16]]
    a1 = [[5, 6, 3, 8], [13, 10, 15, 12]]
    b0 = [[3, 1, 8, 5], [7, 45, 6, 81]]
    b1 = [[7, 1, 8, 2], [11, 34, 5, 71]]
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name,
                                       devices=devices,
                                       num_communicators=num_communicator)
        input0_a = array_ops.constant(a0, dtype=dtypes.int32)
        input0_b = array_ops.constant(b0, dtype=dtypes.int32)
        v0 = comm0.reduce([input0_a, input0_b], root_rank=root_rank)
      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name,
                                       devices=devices,
                                       num_communicators=num_communicator)
        input1_a = array_ops.constant(a1, dtype=dtypes.int32)
        input1_b = array_ops.constant(b1, dtype=dtypes.int32)
        v1 = comm1.reduce([input1_a, input1_b], root_rank=root_rank)
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(use_gpu=True, graph=graph,
                           config=self._config) as sess:
      sess.run(init_op)
      r0, r1 = sess.run([v0, v1])
      if root_rank == 0:
        self.assertAllEqual(r0[0], np.add(a0, a1))
        self.assertAllEqual(r0[1], np.add(b0, b1))
      else:
        self.assertAllEqual(r1[0], np.add(a0, a1))
        self.assertAllEqual(r1[1], np.add(b0, b1))

  def test_reduce_tesors(self):
    for root in [0, 1]:
      for num_communicator in [None, 1]:
        self.reduce_tensors(root_rank=root, num_communicator=num_communicator)

  def reduce_tensors_reuse(self, reduce_op1, reduce_op2, root_rank, num_communicator):
    a0 = [[1, 9, 7, 4], [9, 14, 11, 16]]
    a1 = [[5, 6, 3, 8], [13, 10, 15, 12]]
    b0 = [[3, 1, 8, 5], [7, 45, 6, 81]]
    b1 = [[7, 1, 8, 2], [11, 34, 5, 71]]
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices,
                                       num_communicators=num_communicator)
        input0_a = array_ops.constant(a0, dtype=dtypes.int32)
        input0_b = array_ops.constant(b0, dtype=dtypes.int32)
        v0 = comm0.reduce([input0_a, input0_b],
                          root_rank=root_rank, reduce_op=reduce_op1)
        with ops.control_dependencies(v0):
          v01 = comm0.reduce([input0_a, input0_b],
                             root_rank=root_rank, reduce_op=reduce_op2)
      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices,
                                       num_communicators=num_communicator)
        input1_a = array_ops.constant(a1, dtype=dtypes.int32)
        input1_b = array_ops.constant(b1, dtype=dtypes.int32)
        v1 = comm1.reduce([input1_a, input1_b],
                          root_rank=root_rank, reduce_op=reduce_op1)
        with ops.control_dependencies(v1):
          v11 = comm1.reduce([input1_a, input1_b],
                             root_rank=root_rank, reduce_op=reduce_op2)
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(
        use_gpu=True, graph=graph, config=self._config) as sess:
      sess.run(init_op)
      r0, r01, r1, r11 = sess.run([v0, v01, v1, v11])
      if root_rank == 0:
        y0, y1 = r0, r01
      else:
        y0, y1 = r1, r11
      if reduce_op1 == Communicator.SUM:
        self.assertAllEqual(y0[0], np.add(a0, a1))
        self.assertAllEqual(y0[1], np.add(b0, b1))
      if reduce_op1 == Communicator.MAX:
        self.assertAllEqual(y0[0], np.maximum(a0, a1))
        self.assertAllEqual(y0[1], np.maximum(b0, b1))
      if reduce_op2 == Communicator.SUM:
        self.assertAllEqual(y1[0], np.add(a0, a1))
        self.assertAllEqual(y1[1], np.add(b0, b1))
      if reduce_op2 == Communicator.MAX:
        self.assertAllEqual(y1[0], np.maximum(a0, a1))
        self.assertAllEqual(y1[1], np.maximum(b0, b1))

  def test_reduce_tensors_reuse_root0(self):
    root_rank = 0
    c1 = Communicator.SUM
    c2 = Communicator.MAX
    n = 1
    for op1, op2 in [(c1, c2), (c1, c1), (c2, c2)]:
      self.reduce_tensors_reuse(op1, op2, root_rank, n)

  def test_reduce_tensors_reuse_root1(self):
    root_rank = 1
    c1 = Communicator.SUM
    c2 = Communicator.MAX
    n = 1
    for op1, op2 in [(c1, c2), (c1, c1), (c2, c2)]:
      self.reduce_tensors_reuse(op1, op2, root_rank, n)

  def test_reduce_tensors_reuse_root1_multiple_communicator(self):
    root_rank = 1
    c1 = Communicator.SUM
    c2 = Communicator.MAX
    n = 4
    for op1, op2 in [(c1, c2), (c1, c1), (c2, c2)]:
      self.reduce_tensors_reuse(op1, op2, root_rank, n)

  def reduce_one_tensors_reuse(self, reduce_op1, reduce_op2, root_rank,
                               num_communicator):
    a0 = [[1, 9, 7, 4], [9, 14, 11, 16]]
    a1 = [[5, 6, 3, 8], [13, 10, 15, 12]]
    devices = ['/gpu:0', '/gpu:1']
    shared_name = "comm"
    with ops.Graph().as_default() as graph:
      with ops.device('/gpu:0'):
        comm0 = CollectiveCommunicator(name=shared_name, devices=devices,
                                       num_communicators=num_communicator)
        input0_a = array_ops.constant(a0, dtype=dtypes.int32)
        v0 = comm0.reduce([input0_a], root_rank=root_rank, reduce_op=reduce_op1)
        with ops.control_dependencies(v0):
          v01 = comm0.reduce([input0_a], root_rank=root_rank,
                             reduce_op=reduce_op2)
      with ops.device('/gpu:1'):
        comm1 = CollectiveCommunicator(name=shared_name, devices=devices,
                                       num_communicators=num_communicator)
        input1_a = array_ops.constant(a1, dtype=dtypes.int32)
        v1 = comm1.reduce([input1_a], root_rank=root_rank, reduce_op=reduce_op1)
        with ops.control_dependencies(v1):
          v11 = comm1.reduce([input1_a], root_rank=root_rank,
                             reduce_op=reduce_op2)
      init_op = control_flow_ops.group(
          resources.initialize_resources(resources.local_resources()))
    graph.finalize()

    with self.test_session(
        use_gpu=True, graph=graph, config=self._config) as sess:
      sess.run(init_op)
      r0, r01, r1, r11 = sess.run([v0, v01, v1, v11])
      if root_rank == 0:
        y0, y1 = r0, r01
      else:
        y0, y1 = r1, r11
      if reduce_op1 == Communicator.SUM:
        self.assertAllEqual(y0[0], np.add(a0, a1))
      if reduce_op1 == Communicator.MAX:
        self.assertAllEqual(y0[0], np.maximum(a0, a1))
      if reduce_op2 == Communicator.SUM:
        self.assertAllEqual(y1[0], np.add(a0, a1))
      if reduce_op2 == Communicator.MAX:
        self.assertAllEqual(y1[0], np.maximum(a0, a1))

  def test_reduce_one_tensors_reuse(self):
    for op1 in [Communicator.SUM, Communicator.MAX]:
      for op2 in [Communicator.SUM, Communicator.MAX]:
        for root_rank in [0, 1]:
          for num_communicator in range(1, 4):
            self.reduce_one_tensors_reuse(op1, op2, root_rank, num_communicator)
# pylint: enable=unused-variable,missing-docstring


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  test.main()
