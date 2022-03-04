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
"""Communicator base ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
try:
  from tensorflow.python.training import device_util
except: # pylint: disable=bare-except
  from tensorflow.python.distribute import device_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops

from epl.env import Env

class Communicator(object): # pylint: disable=useless-object-inheritance
  """A communicator with collective operations."""

  SUM = 0
  PROD = 1
  MAX = 2
  MIN = 3
  ALL_REDUCE = 10
  BROADCAST = 11
  REDUCE = 12
  DEFAULT_DEVICE = '/job:localhost'

  @classmethod
  def bcast(
      cls, dtype, shape, fn, devices,
      rank=None, root_rank=0):
    r"""Broadcast values using collective_ops.

    Args:
      dtype: Data type of tensor to broadcast.
      shape: Shape of tensor to broadcast.
      fn: Function to generate tensor for broadcast.
      devices: All devices for communication.
      rank: Current device rank.
      root_rank: Root rank to broadcast.
    """
    current_device = Communicator.current_device()
    world_size = len(devices)
    if rank is None:
      rank = devices.index(current_device)
    local_rank = Env.get().cluster.get_local_rank(rank)
    collective_keys = Env.get().collective_keys[local_rank]
    group_key = collective_keys.get_group_key([current_device])
    collective_instance_key = collective_keys.get_variable_instance_key()
    if root_rank == rank:
      value = fn()
      bcast_send = collective_ops.broadcast_send(
          value, shape, dtype, world_size, group_key, collective_instance_key)
      with ops.control_dependencies([bcast_send]):
        return array_ops.identity(value)
    else:
      return collective_ops.broadcast_recv(
          shape, dtype, world_size, group_key, collective_instance_key)

  @classmethod
  def current_device(cls):
    """Current device."""
    return device_util.canonicalize(
        device_util.current(), default=cls.DEFAULT_DEVICE)

  @classmethod
  def create(cls, shared_name, devices, impl, **kwargs):
    """Create a communicator.

    Args:
      shared_name: shared name of the communicator.
      devices: devices of the communicator.
      impl: implementation class for communication.
      kwargs: (Optional.) key-value arguments.
    """
    if not devices:
      raise ValueError('devices must be provided')
    devices = [
        device_util.canonicalize(
            d.strip(), default=Communicator.DEFAULT_DEVICE) for d in devices]
    if not devices:
      raise ValueError("Devices must not be empty.")
    return impl(shared_name, devices=devices, **kwargs)

  def __init__(self, shared_name, devices):
    """Constructs a communicator instance.

    Args:
      shared_name: shared name of the communicator.
      devices: devices of the communicator.
    """
    if shared_name:
      shared_name = shared_name.replace(':', '_').replace('/', '_')
    else:
      shared_name = ops.get_default_graph().unique_name('communicator')
    self._shared_name = shared_name
    self._device = Communicator.current_device()
    if self._device not in devices:
      raise ValueError(
          "Current device {} not in devices {}".format(
              self._device, devices))
    self._devices = devices

  @property
  def shared_name(self):
    """Shared name of the communicator."""
    return self._shared_name

  @property
  def device(self):
    """Device of the communicator."""
    return self._device

  @property
  def devices(self):
    """All devices of the communicator."""
    return self._devices

  @property
  def size(self):
    """Count of devices."""
    return len(self.devices)

  @property
  def rank(self):
    """Rank of the current device."""
    return self.devices.index(self.device)

  @property
  def scope(self):
    """Scope of the communciator."""
    return '{}/{}'.format(self.shared_name, self.rank)

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

  def reduce(self, value, reduce_op=SUM, root_rank=0):
    """Reduce values across devices to the root device.

    Args:
      value: Value on current device to be reduced and scattered.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.
      root_rank: Rank of broadcast root.

    Returns:
      Reduced and scattered value on current device.
    """
    raise NotImplementedError

  def all_gather(self, value, varying_size=True):
    """Gather all values across devices to all devices.

    Args:
      value: Value on current device to be gathered.
      varying_size: Supposing all value sizes on devices are not equal.

    Returns:
      Gathered value.
    """
    raise NotImplementedError

  def reduce_scatter(self, value, reduce_op=SUM):
    """All reduce values across devices and scatter the result to all devices.

    Args:
      value: Value on current device to be reduced and scattered.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.

    Returns:
      Reduced and scattered value on current device.
    """
    raise NotImplementedError

  def all_reduce(self, value, reduce_op=SUM):
    """All reduce on value across devices.

    Args:
      value: Value to be allreduced.
      reduce_op: Reduction ops: SUM, PROD, MAX or MIN.

    Returns:
      Allreduced value.
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
    raise NotImplementedError

  def all_to_all(self, value, varying_size=True, common_shape=None):
    """alltoall value across devices.

    Args:
      value: Value to be sent to other devices.
      varying_size: whether each message on a rank shall have the same size.
      common_shape: common shape of tensors in value.

    Returns:
      received values from other devices.
    """
    raise NotImplementedError

  def batch_all_to_all(self, values, varying_size=True, common_shapes=None):
    """alltoall batch of values across devices.

    Args:
      values: Batch of values to be sent to other devices.
      varying_size: whether each message on a rank shall have the same size.
      common_shapes: common shapes of tensors in values.

    Returns:
      received batch of values from other devices.
    """
    raise NotImplementedError
