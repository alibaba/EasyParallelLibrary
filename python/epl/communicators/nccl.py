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
"""NCCL communication ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops

from epl.communicators.base import Communicator
from epl.communicators import nccl_ops

class NcclCommunicator(Communicator):
  """A communicator using NCCL."""

  class Resource(object): # pylint: disable=useless-object-inheritance
    """Resource object of a communciator."""
    def __init__(self, comm):
      self._comm = comm

    @property
    def name(self):
      """Resource name of the communicator."""
      return self._comm.name

    @property
    def handle(self):
      """Resource handle of the communicator."""
      return self._comm._handle.op  # pylint: disable=protected-access

    @property
    def create(self):
      """Resource creation op of the communicator."""
      return self._comm._create_op  # pylint: disable=protected-access

    @property
    def is_initialized(self):
      """Resource creation check op of the communicator."""
      return self._comm._is_initialized_op  # pylint: disable=protected-access

  def __init__(self, shared_name, devices, chief_rank=0,
               root_rank=0, **kwargs):
    """Constructs a NCCL communicator instance.

    Args:
      shared_name: shared name of the communicator.
      devices: devices of the communicator.
      chief_rank: (Optional.) Chief device rank for the communicator.
      root_rank: The rank to communicate using tier 1.
      kwargs: (Optional.) key-value arguments.
    """
    super(NcclCommunicator, self).__init__(shared_name, devices)
    self._chief_rank = chief_rank

    if self.size == 1:
      return

    for d in self.devices:
      if 'GPU' not in d:
        raise ValueError('NCCL is only supported by GPU')

    self._root_rank = root_rank
    self._kwargs = kwargs

    ops.add_to_collection(
        ops.GraphKeys.LOCAL_RESOURCES, self.build_resource())

  @property
  def name(self):
    return '{}/replicas/{}'.format(self.shared_name, self.rank)

  @property
  def chief_rank(self):
    return self._chief_rank

  @property
  def root_rank(self):
    return self._root_rank

  def build_resource(self):
    """Create NcclCommunicator.Resource for flat network."""
    with ops.name_scope(self.scope):
      with ops.control_dependencies(None):
        nccl_id = nccl_ops.communicator_get_id_broadcast(
            devices=self.devices, root_rank=self.chief_rank, rank=self.rank)
        self._handle = nccl_ops.NcclCommunicatorHandle(
            self.shared_name, root_rank=0)
        self._create_op = self._handle.create(
            nccl_id, size=self.size, rank=self.rank)
        self._is_initialized_op = self._handle.is_initialized()
        return NcclCommunicator.Resource(self)

  def reduce(self, value, reduce_op=Communicator.SUM, root_rank=0):
    """Reduce values across devices to the root device.

    Args:
      value: Value on current device to be reduced.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.
      root_rank: Rank of reduce root.

    Returns:
      Reduced value.
    """
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      return self._handle.reduce(value, reduce_op, root_rank)

  def gather(self, value, root_rank=0, varying_size=True):
    """Gather all values across devices to root device.

    Args:
      value: Value on current device to be gathered.
      root_rank: Rank of root gather device.
      varying_size: Supposing all value sizes on devices are not equal.

    Returns:
      Gathered value on root, None or controlling object on non-root.
    """
    raise NotImplementedError

  def scatter(self, value, root_rank=0):
    """Scatter value on root device to all devices.

    Args:
      value: Value on current device to be scattered, no use for non-root.
      root_rank: Rank of root scatter device.

    Returns:
      Scattered value on current device.
    """
    raise NotImplementedError

  def broadcast(self, value, root_rank=0):
    """Broadcast value across devices.

    Args:
      value: Value to broadcast.
      root_rank: Rank of broadcast root.

    Returns:
      Broadcasted value.
    """
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      return self._handle.broadcast(value, root_rank)

  def all_reduce(self, value, reduce_op=Communicator.SUM):
    """All reduce on value across devices using NCCL.

    Args:
      value: Value to be allreduced.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.

    Returns:
      Allreduced value.
    """
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      return self._handle.all_reduce(value, reduce_op)

  def all_gather(self, value, varying_size=True):
    """Gather all values across devices to all devices.

    Args:
      value: Value on current device to be gathered.
      varying_size: Supposing all value sizes on devices are not equal.

    Returns:
      Gathered value.
    """
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      return self._handle.all_gather(value, varying_size)

  def reduce_scatter(self, value, reduce_op=Communicator.SUM):
    """All reduce values across devices and scatter the result to all devices.

    Args:
      value: Value on current device to be reduced and scattered.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.

    Returns:
      Reduced and scattered value on current device.
    """
    if self.size == 1:
      return value

    with ops.name_scope(self.scope):
      return self._handle.reduce_scatter(value, reduce_op)

  def all_to_all(self, value, varying_size=True, common_shape=None):
    """alltoall value across devices.

    Args:
      value: Value to send to other devices.
      varying_size: whether each message on a rank shall have the same size.
      common_shape: common shape of tensors in value.

    Returns:
      received values from other devices.
    """
    if self.size == 1:
      return value
    if common_shape is None:
      common_shape = {}

    with ops.name_scope(self.scope):
      return self._handle.all_to_all(
          value, varying_size=varying_size, common_shape=common_shape)
