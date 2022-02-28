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
"""Rewriter to sum sparse gradients across devices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops

from epl.communicators.options import build_communicator
from epl.communicators.options import CommunicatorSpec
from epl.communicators.rewriters.base import Rewriter

def _chunk_list(items, num_chunks):
  '''Split the list, items, into num_chunks evenly sized chunks'''
  num_items = len(items)
  if num_items == 0:
    return []
  s, r = divmod(num_items, num_chunks)
  t = s + 1
  return ([items[p:p+t] for p in xrange(0, r*t, t)] +
          [items[p:p+s] for p in xrange(r*t, num_items, s)])

class SparseAllReduceRewriter(Rewriter):
  """Rewriter to sum sparse gradients across devices.

  NOTE: This rewriter should be called before rewriters for dense tensors
        (e.g. CoalescingRewriter).
  """
  def __init__(self, enabled=False, enable_fp16=False, fp16_scale=None,
               num_chunks=None, chunk_overlap=False,
               comm_impl=None, **comm_kwargs):
    """Create a rewriter to sum sparse gradients across devices.

    Args:
      enabled: (Optional.) True if enabled.
      enable_fp16: (Optional.) True if enabled.
      fp16_scale: (Optional.) Scale the values before sparse allreduce.
      comm_impl: (Optional.) class of communicator for sparse allreduce.
      comm_kwargs: (Optional.) arguments of communicator for sparse allreduce.
    """
    self._enabled = enabled
    self._num_chunks = num_chunks
    self._chunk_overlap = chunk_overlap
    self._enable_fp16 = enable_fp16
    self._fp16_scale = fp16_scale
    self._comm_impl = comm_impl
    self._comm_kwargs = comm_kwargs

  @property
  def enabled(self):
    """True if enabled."""
    return self._enabled

  @property
  def num_chunks(self):
    """Number of comms."""
    return self._num_chunks

  @property
  def chunk_overlap(self):
    """Overlap values and indices in one chunk."""
    return self._chunk_overlap

  def rewrite(self, _, fn, values, comm_name, comm_spec, **kwargs):
    """Call function on specific device.

    Args:
      fn: values, comm_name, comm_spec, **kwargs -> value.
      values: a value or list of values.
      comm_name: unique name for this rewriter.
      comm_spec: communicator builder.
      kwargs: (Optional.) key-value arguments of fn.

    Returns:
      result values.
    """
    if not self.enabled:
      return fn(values, comm_name, comm_spec, **kwargs)

    result_values = [None] * len(values)
    sparse_tensors = []
    sparse_tensor_indices = []
    dense_tensors = []
    dense_tensor_indices = []
    for idx, v in enumerate(values):
      if isinstance(v, ops.IndexedSlices):
        if self._enable_fp16:
          sparse_tensors.append(self._convert_to_fp16(v, self._fp16_scale))
        else:
          sparse_tensors.append(v)
        sparse_tensor_indices.append(idx)
      else:
        dense_tensors.append(v)
        dense_tensor_indices.append(idx)

    result_sparse_tensors = []
    sparse_allreduce_comm_kwargs = dict(comm_spec.kwargs)
    sparse_allreduce_comm_kwargs.update(self._comm_kwargs)

    sparse_allreduce_comm_spec = CommunicatorSpec(
        devices=comm_spec.devices,
        comm_impl=self._comm_impl or comm_spec.impl,
        **sparse_allreduce_comm_kwargs)

    num_chunks = self.num_chunks if self.num_chunks else 1
    if num_chunks > len(sparse_tensors):
      num_chunks = len(sparse_tensors)
    if num_chunks < 1:
      num_chunks = 1
    sparse_tensors_chunks = _chunk_list(sparse_tensors, num_chunks)
    for chunk_id, chunk in enumerate(sparse_tensors_chunks):
      sparse_allreduce_comm = build_communicator(
          '{}_sparse_allreduce_chunk_{}'.format(comm_name, chunk_id),
          sparse_allreduce_comm_spec)
      if self.chunk_overlap:
        sparse_indices_allreduce_comm = build_communicator(
            '{}_sparse_indices_allreduce_chunk_{}'.format(comm_name, chunk_id),
            sparse_allreduce_comm_spec)
      sparse_allreduce_ops = []
      for idx, t in enumerate(chunk):
        if idx == 0:
          reduced_values = sparse_allreduce_comm.all_gather(t.values)
          if self.chunk_overlap:
            reduced_indices = sparse_indices_allreduce_comm.all_gather(
                t.indices)
          else:
            with ops.control_dependencies([reduced_values]):
              reduced_indices = sparse_allreduce_comm.all_gather(t.indices)
          sparse_allreduce_ops.append([reduced_values, reduced_indices])
        else:
          with ops.control_dependencies(sparse_allreduce_ops[-1]):
            reduced_values = sparse_allreduce_comm.all_gather(t.values)
            if self.chunk_overlap:
              reduced_indices = sparse_indices_allreduce_comm.all_gather(
                  t.indices)
            else:
              with ops.control_dependencies([reduced_values]):
                reduced_indices = sparse_allreduce_comm.all_gather(t.indices)
            sparse_allreduce_ops.append([reduced_values, reduced_indices])
        result_sparse_tensors.append(
            ops.IndexedSlices(
                reduced_values,
                reduced_indices,
                dense_shape=t.dense_shape))

    for idx, t in enumerate(result_sparse_tensors):
      if self._enable_fp16:
        result_values[sparse_tensor_indices[idx]] = \
          self._convert_to_fp32(t, self._fp16_scale)
      else:
        result_values[sparse_tensor_indices[idx]] = t

    result_dense_tensors = fn(dense_tensors, comm_name, comm_spec, **kwargs)
    for idx, t in enumerate(result_dense_tensors):
      result_values[dense_tensor_indices[idx]] = t

    return result_values
