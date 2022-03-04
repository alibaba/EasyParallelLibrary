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
# ==============================================================================
"""NCCL ops wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.framework import tensor_shape

from epl.communicators.base import Communicator
from epl.communicators.pywrap import _ops

ops.NotDifferentiable('EplNcclCommunicatorGetId')

ops.NotDifferentiable('EplNcclCommunicatorHandleOp')
ops.NotDifferentiable('EplNcclCommunicatorCreater')
ops.NotDifferentiable('EplNcclCommunicatorIsInitialized')

ops.NotDifferentiable('EplNcclCommunicatorBroadcast')

@ops.RegisterGradient('EplNcclCommunicatorAllReduce')
def _epl_nccl_communicator_all_reduce_grad(op, grad):
  """Gradient for NCCL allreduce op.
  """
  comm_handle = op.inputs[0]
  reduce_op = op.get_attr("reduce_op")
  if reduce_op != Communicator.SUM:
    raise NotImplementedError(
        "Error occured during AllReduce auto diff. Only differential of "
        "reduce SUM operation could be supported.")
  with ops.device(op.device):
    with ops.control_dependencies([op]):
      grad = _ops.epl_nccl_communicator_all_reduce(
          comm_handle, grad, reduce_op=reduce_op)
  return [None, grad]

@ops.RegisterGradient('EplNcclCommunicatorAllGather')
def _epl_nccl_communicator_all_gather_grad(op, grad):
  """Gradient for NCCL allgather op.
  """
  comm_handle = op.inputs[0]
  with ops.device(op.device):
    with ops.control_dependencies([op]):
      grad = _ops.epl_nccl_communicator_reduce_scatter(
          comm_handle, grad, reduce_op=Communicator.SUM)
  return [None, grad]

ops.NotDifferentiable('EplNcclCommunicatorAllGatherv')

@ops.RegisterGradient('EplNcclCommunicatorReduceScatter')
def _epl_nccl_communicator_reduce_scatter_grad(op, grad):
  """Gradient for NCCL reduce scatter op.
  """
  comm_handle = op.inputs[0]
  reduce_op = op.get_attr("reduce_op")
  if reduce_op != Communicator.SUM:
    raise NotImplementedError(
        "Error occured during AllReduce auto diff. Only differential of "
        "reduce SUM operation could be supported.")
  with ops.device(op.device):
    with ops.control_dependencies([op]):
      grad = _ops.epl_nccl_communicator_all_gather(comm_handle, grad)
  return [None, grad]

@ops.RegisterGradient('EplNcclCommunicatorReduce')
def _epl_nccl_communicator_reduce_grad(op, grad):
  """Gradient for NCCL reduce op.
  """
  comm_handle = op.inputs[0]
  reduce_op = op.get_attr("reduce_op")
  rank = op.get_attr("rank")
  root_rank = op.get_attr("root_rank")
  if reduce_op != Communicator.SUM:
    raise NotImplementedError(
        "Error occured during Reduce auto diff. Only differential of "
        "reduce SUM operation could be supported.")
  with ops.device(op.device):
    with ops.control_dependencies([op]):
      grad = _ops.epl_nccl_communicator_broadcast(
          comm_handle, grad, root_rank=root_rank, rank=rank)
  return [None, grad]

@ops.RegisterGradient('EplNcclCommunicatorAllToAll')
def _epl_nccl_communicator_all_to_all_grad(op, *args):
  """Gradient for NCCL all to all op.
  """
  comm_handle = op.inputs[0]
  rank = op.get_attr("rank")
  grad_in = args[0]
  with ops.device(op.device):
    with ops.control_dependencies([op]):
      grad_out = _ops.epl_nccl_communicator_all_to_all(
          comm_handle, grad_in, rank=rank)
  return [None, grad_out]

@ops.RegisterGradient('EplNcclCommunicatorAllToAllv')
def _epl_nccl_communicator_all_to_all_v_grad(op, *args):
  """Gradient for NCCL all to all op.
  """
  comm_handle = op.inputs[0]
  rank = op.get_attr("rank")
  common_shape = op.get_attr("common_shape")
  grad_in = list(args)
  with ops.device(op.device):
    with ops.control_dependencies([op]):
      grad_out = _ops.epl_nccl_communicator_all_to_allv(
          comm_handle, grad_in, rank=rank, common_shape=common_shape)
  return [None] + grad_out

def communicator_get_id_broadcast(devices, root_rank, rank):
  def _get_id():
    return _ops.epl_nccl_communicator_get_id()
  return Communicator.bcast(
      dtypes.int64, tensor_shape.TensorShape([16]), # 128 / 8
      _get_id, devices, rank=rank, root_rank=root_rank)

class NcclCommunicatorHandle(object): # pylint: disable=useless-object-inheritance
  """Wrapper of a NCCL communicator handle."""
  def __init__(self, shared_name=None, root_rank=0):
    self._shared_name = shared_name
    self._root_rank = root_rank
    self._handle = _ops.epl_nccl_communicator_handle_op(
        shared_name=shared_name)

  @property
  def size(self):
    return self._size

  @property
  def rank(self):
    return self._rank

  @property
  def root_rank(self):
    return self._root_rank

  @property
  def shared_name(self):
    return self._shared_name

  @property
  def op(self):
    return self._handle

  def create(self, nccl_id, size, rank):
    """Resource creation op of the communicator."""
    self._size = size
    self._rank = rank
    return _ops.epl_nccl_communicator_creater(
        self._handle, nccl_id, size=size, rank=rank,
        shared_name=self.shared_name)

  def is_initialized(self):
    return _ops.epl_nccl_communicator_is_initialized(self._handle)

  def broadcast(self, value, root_rank):
    with ops.device(value.device):
      return _ops.epl_nccl_communicator_broadcast(
          self._handle, value, root_rank=root_rank,
          size=self.size, rank=self.rank)

  def all_reduce(self, value, reduce_op):
    with ops.device(value.device):
      return _ops.epl_nccl_communicator_all_reduce(
          self._handle, value, reduce_op=reduce_op,
          size=self.size, rank=self.rank)

  def all_gather(self, value, varying_size):
    with ops.device(value.device):
      if varying_size:
        return _ops.epl_nccl_communicator_all_gatherv(
            self._handle, value, size=self.size, rank=self.rank)
      return _ops.epl_nccl_communicator_all_gather(
          self._handle, value, size=self.size, rank=self.rank)

  def reduce_scatter(self, value, reduce_op):
    with ops.device(value.device):
      return _ops.epl_nccl_communicator_reduce_scatter(
          self._handle, value, reduce_op=reduce_op,
          size=self.size, rank=self.rank)

  def reduce(self, value, reduce_op, root_rank=0):
    """reduce values across devices."""
    with ops.device(value.device):
      return _ops.epl_nccl_communicator_reduce(
          self._handle, value, reduce_op=reduce_op, root_rank=root_rank,
          size=self.size, rank=self.rank)

  def all_to_all(self, values, varying_size=True, common_shape=None):
    """alltoall value across devices."""
    if not varying_size:
      if not isinstance(values, (ops.Tensor, variables.RefVariable)):
        raise ValueError("Tensor or Variable allowed for varying_size=False.")
      value = values
      # Expect tensor with 2 dims to all_to_all.
      expected_least_dims = 2
      if value.get_shape().ndims < expected_least_dims or \
         value.get_shape()[0].value % self.size != 0:
        raise ValueError("Expect tensor with at least 2 dims to all_to_all " +
                         "and the first dim should be divided evenly by "
                         "devices. Tensor shape {}, devices number {}.". \
                         format(value.get_shape(), self.size))
      with ops.device(value.device):
        return _ops.epl_nccl_communicator_all_to_all(
            self._handle, value, rank=self.rank)

    if len({v.device for v in values}) > 1:
      raise ValueError('inputs must be placed at same device')
    if len(values) != self.size:
      raise ValueError('Number of inputs must be same to devices')
    if common_shape is None:
      common_shape = {}
    with ops.device(values[0].device):
      return _ops.epl_nccl_communicator_all_to_allv(
          self._handle, values, rank=self.rank, common_shape=common_shape)
