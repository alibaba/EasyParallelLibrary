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
"""Layer to aggregate outputs from one taskgraph to another."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from epl.env import Env
from epl.parallel.ops import create_simple_communicator


class BaseBridgingLayer(object):
  """Base bridging layer to bridge layers with different parallel types."""
  def __init__(self, name):
    self.name = name

  @staticmethod
  def __call__(tensors):
    raise NotImplementedError


class Replica2Replica(BaseBridgingLayer):
  def __call__(self, tensors):
    raise NotImplementedError


class Split2Split(BaseBridgingLayer):
  def __call__(self, tensors):
    raise NotImplementedError


class Replica2Split(BaseBridgingLayer):
  """Bridging taskgraph with replica type and split type."""
  def __call__(self, tensors):
    if tensors is None:
      return tensors
    # Currently only support fuse mode for split strategy,
    # so simply the all devices from first virtual device.
    devices = Env.get().cluster.virtual_devices[0].all_devices
    comm = create_simple_communicator(name="Replica2Split_AllGather_" +
                                      self.name,
                                      devices=devices)
    comm_tensor = comm.allgather(tensors)
    return comm_tensor
