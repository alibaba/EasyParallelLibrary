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
"""Functions for communication."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import math_ops
from epl.communicators.base import Communicator
from epl.communicators.nccl import NcclCommunicator
from epl.communicators.options import build_communicator
from epl.communicators.options import CommunicatorSpec
from epl.communicators.options import Options
from epl.communicators.rewriters.coalescing import CoalescingRewriter
from epl.communicators.rewriters.sparse_allreduce import SparseAllReduceRewriter
from epl.communicators.communication_pool import CommunicationPool
from epl.utils import constant


class CollectiveCommunicator(object):
  """Communicator."""
  def __init__(self, **kwargs):
    self._deepcopy_kwargs = kwargs
    self._options = Options(kwargs,
                            name=None,
                            devices=None,
                            comm_impl=NcclCommunicator,
                            max_splits=None,
                            issorted=True,
                            enable_sparse_allreduce=True,
                            sparse_allreduce_chunks=1,
                            sparse_chunk_overlap=False,
                            num_communicators=None,
                            enable_fp16=False,
                            fp16_scale=None,
                            root_rank=0)

    if self._options.num_communicators is None:
      self._options.num_communicators = 1
    if self._options.max_splits is None:
      self._options.update(max_splits=6)

    # Dict to reuse comm/comm_pool
    self.name2comm = {}

    # Maybe get devices from ENV.
    if not self._options.devices:
      raise ValueError("Devices must not be empty.")

  def get_or_create_comm(self, comm_name, comm_spec, communication_op=None, comm_creator=None):
    """Create comm if not exist."""
    if comm_creator:
      use_pool = False
    else:
      use_pool = True
    comm_key = comm_name + "," + Communicator.current_device()
    if comm_key in self.name2comm:
      comm = self.name2comm[comm_key]
      if use_pool:
        comm.comm_spec.kwargs.update(comm_spec.kwargs)
    else:
      if not use_pool:
        comm = comm_creator()
      else:
        comm = CommunicationPool(self.options.num_communicators,
                                 comm_name,
                                 comm_spec,
                                 communication_op)
      self.name2comm[comm_key] = comm
    return comm

  def __deepcopy__(self, memo):
    kwargs = self._deepcopy_kwargs
    if kwargs:
      kwargs = dict(kwargs)
    copy = CollectiveCommunicator(**kwargs)
    memo[id(self)] = copy
    return copy

  def batch_allreduce(self, tensors, mean=False):
    """Support allreduce for dense tensors, allgather for sparse tensors."""
    comm_spec = CommunicatorSpec(devices=self.options.devices,
                                 comm_impl=self.options.comm_impl)
    comm_name = self.options.name + "_batch_allreduce_pool"
    comm_pool = self.get_or_create_comm(comm_name, comm_spec, communication_op=Communicator.ALL_REDUCE)
    @SparseAllReduceRewriter(
        enabled=self.options.enable_sparse_allreduce,
        num_chunks=self.options.sparse_allreduce_chunks,
        chunk_overlap=self.options.sparse_chunk_overlap,
        enable_fp16=self.options.enable_fp16,
        fp16_scale=self.options.fp16_scale,
        root_rank=self.options.root_rank)
    @CoalescingRewriter(max_splits=self.options.max_splits,
                        issorted=self.options.issorted,
                        enable_fp16=self.options.enable_fp16,
                        fp16_scale=self.options.fp16_scale,
                        enable_logging=True,
                        comm_pool=comm_pool)
    def communicate(values, comm_name, comm_spec):
      """Sum across devices."""
      return comm_pool.communicate(values, comm_name, comm_spec)
    aggregated = communicate(tensors, comm_name, comm_spec)

    if mean:
      num_devices = len(self.options.devices)
      if isinstance(aggregated, (list, tuple)):
        aggregated = [math_ops.cast(v / num_devices, dtype=v.dtype) for v in aggregated]
      else:
        aggregated /= num_devices
    return aggregated

  def broadcast(self, tensors, root_rank=0):
    """Support broadcast tensors from root_rank."""
    comm_spec = CommunicatorSpec(devices=self.options.devices,
                                 comm_impl=self.options.comm_impl,
                                 root_rank=root_rank)
    comm_name = self.options.name + "_broadcast_pool"
    comm_pool = self.get_or_create_comm(comm_name, comm_spec, communication_op=Communicator.BROADCAST)

    @CoalescingRewriter(
        max_splits=self.options.max_splits,
        issorted=self.options.issorted,
        enable_logging=True,
        comm_pool=comm_pool)
    def communicate(values, comm_name, comm_spec):
      """Broadcast from root."""
      return comm_pool.communicate(values, comm_name, comm_spec)
    return communicate(tensors, comm_name, comm_spec)

  def allgather(self, tensors):
    """All gather tensors."""
    comm_name = self.options.name + "_allgather"
    comm_spec = CommunicatorSpec(devices=self.options.devices, comm_impl=self.options.comm_impl)
    comm_fn = lambda: build_communicator(comm_name, comm_spec)
    comm = self.get_or_create_comm(comm_name, comm_spec, comm_creator=comm_fn)
    return comm.all_gather(tensors, varying_size=False)

  def alltoall(self, tensor):
    """All to all tensors."""
    comm_name = self.options.name + "_alltoall"
    comm_spec = CommunicatorSpec(devices=self.options.devices, comm_impl=self.options.comm_impl)
    comm_fn = lambda: build_communicator(comm_name, comm_spec)
    comm = self.get_or_create_comm(comm_name, comm_spec, comm_creator=comm_fn)
    return comm.all_to_all(tensor, varying_size=False)

  def reduce(self, tensors, root_rank=0, reduce_op=Communicator.SUM):
    """Support reduce tensors to root_rank device."""
    comm_spec = CommunicatorSpec(devices=self.options.devices,
                                 comm_impl=self.options.comm_impl,
                                 reduce_op=reduce_op,
                                 root_rank=root_rank)
    comm_name = self.options.name + "_reduce_pool"
    comm_pool = self.get_or_create_comm(comm_name, comm_spec, communication_op=Communicator.REDUCE)

    @CoalescingRewriter(
        max_splits=self.options.max_splits,
        issorted=self.options.issorted,
        enable_logging=True,
        comm_pool=comm_pool)
    def communicate(values, comm_name, comm_spec):
      """Reduce values to root."""
      return comm_pool.communicate(values, comm_name, comm_spec)

    return communicate(tensors, comm_name, comm_spec)

  @property
  def options(self):
    return self._options

def estimate_split_num_for_comm(tensors):
  """Estimate num_splits for tensors using for comm."""
  # count tensors by dtype.
  if not isinstance(tensors, list):
    tensors_list = [tensors]
  else:
    tensors_list = tensors
  count_by_dtype = {}
  for t in tensors_list:
    if t.shape.num_elements():
      if t.dtype in count_by_dtype:
        count_by_dtype[t.dtype] = count_by_dtype[t.dtype] + \
                                  t.shape.num_elements()
      else:
        count_by_dtype[t.dtype] = t.shape.num_elements()

  # estimate num_splits
  num_splits = 0
  for k in count_by_dtype:
    num_dtype = (count_by_dtype[k] * k.size + constant.DEFAULT_COM_SPLIT_SIZE - 1) // constant.DEFAULT_COM_SPLIT_SIZE
    num_splits = num_splits + num_dtype
  return num_splits if num_splits > 1 else 1
