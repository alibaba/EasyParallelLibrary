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
"""Create communication poll for communication_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops

from epl.communicators.base import Communicator
from epl.communicators.options import build_communicator

class CommunicationPool(object):
  """Communication pool."""
  def __init__(self, num_communicators, comm_name, comm_spec,
               communication_op=Communicator.ALL_REDUCE):
    """Create CommunicationPool with some communicators.

    Args:
      num_communicators: number of comunicators.
      comm_name: unique name for communicator.
      comm_spec: communicator builder.
      communication_op: communication op, such as allreduce, broadcast, reduce.
      kwargs: (Optional.) key-value arguments.
    """
    self._num_communicators = num_communicators
    self._communication_op = communication_op
    self._communicator_list = [
        build_communicator('{}/group_{}'.format(comm_name, index), comm_spec)
        for index in range(num_communicators)]
    self._comm_spec = comm_spec
    self._comm_name = comm_name

  @property
  def num_communicators(self):
    """Number of communicators."""
    return self._num_communicators

  @property
  def comm_name(self):
    """Name of communicators."""
    return self._comm_name

  @property
  def comm_spec(self):
    """Spec of communicators."""
    return self._comm_spec

  @property
  def communication_op(self):
    """Communication op."""
    return self._communication_op

  def _communication_fn(self, communicator, flattened):
    """Communication function."""
    if self.communication_op == Communicator.BROADCAST:
      root_rank = self.comm_spec.kwargs.get("root_rank", 0)
      return communicator.broadcast(flattened, root_rank=root_rank)
    elif self.communication_op == Communicator.ALL_REDUCE:
      return communicator.all_reduce(flattened, reduce_op=Communicator.SUM)
    elif self.communication_op == Communicator.REDUCE:
      root_rank = self.comm_spec.kwargs.get("root_rank", 0)
      reduce_op = self.comm_spec.kwargs.get("reduce_op", Communicator.SUM)
      return communicator.reduce(flattened, reduce_op=reduce_op,
                                 root_rank=root_rank)
    else:
      raise TypeError("Unsupport communication op: {}".format(
          self.communication_op))

  # pylint: disable=unused-argument
  def communicate(self, buffers, **kwargs):
    """"Do communication for buffers."""
    tupid = 0
    reduced_buffers = []
    dep_list = {}
    # Reverse the buffers for the order of gradients.
    for flattened in buffers[::-1]:
      comm_id = tupid % self.num_communicators
      is_first = (tupid // self.num_communicators) == 0
      if is_first:
        last_values = self._communication_fn(self._communicator_list[comm_id],
                                             flattened)
        dep_list[comm_id] = last_values
      else:
        with ops.control_dependencies([dep_list[comm_id]]):
          last_values = self._communication_fn(self._communicator_list[comm_id],
                                               flattened)
          dep_list[comm_id] = last_values
      # Recovery order for reduced_buffers.
      reduced_buffers.insert(0, last_values)
      tupid = tupid + 1
    return reduced_buffers
  # pylint: enable=unused-argument
