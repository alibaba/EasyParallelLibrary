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
"""Rewriter to split communication to buffers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os
from six.moves import xrange # pylint: disable=redefined-builtin

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging

from epl.communicators.rewriters.base import Rewriter

def estimate_ticks(tensors, issorted=False):
  """Estimate execution time of input tensors.

  Args:
    tensors: tensors to estimate.
    issorted: tensors already sorted.

  Returns:
    Estimated execution time of input tensors.
  """
  if issorted:
    return list(xrange(len(tensors)))

  ticks = []
  op_tick_dict = {}
  def lookup_op_tick(op):
    '''Lookup tick of input op.'''
    if op in op_tick_dict:
      return op_tick_dict[op]
    wave = set([op])
    waves = []
    ticks = [0]
    while wave:
      waves.append(wave)
      ticks[0] += 1
      next_wave = set()
      for o in wave:
        if not o.inputs:
          continue
        if o in op_tick_dict:
          ticks.append(ticks[0] + op_tick_dict[o] - 1)
          continue
        next_wave.update({t.op for t in o.inputs})
      wave = next_wave
    op_tick_dict[op] = max(ticks)
    next_tick = 0
    for wave in reversed(waves):
      tick = 0
      for o in wave:
        if not o.inputs:
          tick = max(tick, 1)
        elif o in op_tick_dict:
          tick = max(tick, op_tick_dict[o])
        else:
          op_tick_dict[o] = 1 + next_tick
          tick = max(tick, op_tick_dict[o])
      next_tick = tick
    return op_tick_dict[op]

  for tid, t in enumerate(tensors):
    if isinstance(t, ops.IndexedSlices):
      ticks.append(lookup_op_tick(t.values.op))
    else:
      ticks.append(lookup_op_tick(t.op))
    logging.vlog(
        1, 'Tensor {} ({}/{}) estimated.'.format(t.name, tid + 1, len(tensors)))
  return ticks

def sort_by_tick(tensors, issorted=False):
  """Sort tensors by estimated ticks.

  Args:
    tensors: tensors to sort.
    issorted: tensors already sorted.

  Returns:
    sorted_tensors: sorted tensors.
    sorted_ticks: ticks of sorted tensors.
    sorted_indices: indices of sorted tensors.
  """
  tensor_dtypes = [repr(t.dtype) for t in tensors]
  estimated_ticks = estimate_ticks(tensors, issorted=issorted)
  tensor_indices = list(xrange(len(tensors)))
  sorted_tensor_tuples = sorted(zip(
      tensor_dtypes, estimated_ticks, tensor_indices, tensors))
  _, sorted_ticks, sorted_indices, sorted_tensors = zip(*sorted_tensor_tuples)
  return sorted_tensors, sorted_ticks, sorted_indices

def unsort_to_tensors(tensors, sorted_indices):
  """Get original tensors from sorted tensors.

  Args:
    tensor: sorted tensors.
    sorted_indices: indices of sorted tensors.

  Returns:
    original tensors.
  """
  return list(zip(*sorted(zip(sorted_indices, tensors))))[1]

def split_to_tuples(sorted_tensors, sorted_ticks, max_splits=1):
  """Split sorted tensors into tuples.

  Args:
    sorted_tensors: sorted tensors to split.
    sorted_ticks: ticks of sorted tensors.
    max_splits: max number of splits.

  Returns:
    list of tensor tuples.
  """

  if not sorted_tensors:
    return []

  # Fast path for single tensor.
  if len(sorted_tensors) == 1:
    return [sorted_tensors]

  # Pass 1: split by dtypes.
  dtype_tuples = []
  dtype_ticks = []
  prev_dtype = None
  for tensorid, t in enumerate(sorted_tensors):
    if t.dtype != prev_dtype:
      dtype_tuples.append([t])
      dtype_ticks.append([sorted_ticks[tensorid]])
      prev_dtype = t.dtype
    else:
      dtype_tuples[-1].append(t)
      dtype_ticks[-1].append(sorted_ticks[tensorid])
  dtype_num = len(dtype_tuples)
  if dtype_num > max_splits:
    logging.info(
        'The number of dtypes ({ndtypes}) > max_splits({nsplits}), split '
        'tensors into {ndtypes} tuples.'.format(
            ndtypes=dtype_num, nsplits=max_splits))
    return dtype_tuples
  if dtype_num == max_splits:
    return dtype_tuples

  # Pass 2: split by sizes.
  # Pass 2.1: Calculate number of splits for different dtypes.
  dtype_tuple_scores = [t[-1] - t[0] for t in dtype_ticks]
  dtype_tuple_sum = sum(dtype_tuple_scores)
  if dtype_tuple_sum <= 0:
    dtype_tuple_sum = 1
  tuple_num_splits = [
      int(max_splits * s / dtype_tuple_sum) for s in dtype_tuple_scores]
  tuple_num_splits = [n if n > 0 else 1 for n in tuple_num_splits]
  tuple_num_splits[0] += (max_splits - sum(tuple_num_splits))

  tuples = []
  for tupid, tup in enumerate(dtype_tuples):
    # Pass 2.2: For each dtype, calculate split size.
    tup_meta = [(t.dtype.size, t.shape.num_elements()) for t in tup]
    tup_sizes_or_none = [s * n if n else None for s, n in tup_meta]
    tup_non_empty_sizes = [s for s in tup_sizes_or_none if s]
    sum_tup_non_empty_sizes = sum(tup_non_empty_sizes)
    mean_size = sum_tup_non_empty_sizes / len(tup_non_empty_sizes)
    tup_sizes = [s if s else mean_size for s in tup_sizes_or_none]
    if tuple_num_splits[tupid] == 1:
      split_size = sum(tup_sizes)
    else:
      split_size = sum(tup_sizes) / (tuple_num_splits[tupid] - 1)

    # Pass 2.3: For each dtype, split tuples by split size.
    tup_tuples = []
    prev_size = 0
    for tensorid, t in enumerate(tup):
      if prev_size == 0 or prev_size + tup_sizes[tensorid] > split_size:
        tup_tuples.append([t])
        prev_size = tup_sizes[tensorid]
      else:
        tup_tuples[-1].append(t)
        prev_size += tup_sizes[tensorid]
    tuples.extend(tup_tuples)
  tuples = [t for t in tuples if t]
  return tuples

def unsplit_to_tensors(tuples):
  """Get original tensor list from list of tensor tuples.

  Args:
    tuples: list of tensor tuples.

  Returns:
    original tensor list.
  """
  return [t for tup in tuples for t in tup]

def flatten_to_buffer(tensors):
  """Flatten list of tensors into one buffer tensor.

  Args:
    tensors: tensors to flatten.

  Returns:
    flattened: flattened buffer tensor.
    shapes: shapes of tensors.
    sizes: sizes of tensors.
  """
  ftensors = [array_ops.reshape(t, [-1]) for t in tensors]
  shapes = [array_ops.shape(t) for t in tensors]
  sizes = [array_ops.reshape(array_ops.shape(t), []) for t in ftensors]
  return array_ops.concat(ftensors, 0), shapes, sizes

def deflatten_to_tensors(flattened, shapes, sizes):
  """Get original tensor list from flattened buffer tensor.

  Args:
    flattened: flattened buffer tensor.
    shapes: shapes of tensors.
    sizes: sizes of tensors.

  Returns:
    original tensor list.
  """
  ftensors = array_ops.split(flattened, sizes)
  return [array_ops.reshape(t, shapes[i]) for i, t in enumerate(ftensors)]

def tensors_debug_string(tensors):
  """Debug string of tensors.

  Args:
    tensors: tensor list.

  Returns:
    debug string of tensor list.
  """
  dtypes = [t.dtype for t in tensors]
  sizes = [
      t.dtype.size * t.shape.num_elements()
      if t.shape.num_elements() is not None and t.shape.num_elements() > 0
      else None
      for t in tensors]
  nonempty_sizes = [s for s in sizes if s]
  return "{} tensors ({}): {:.2f} MB and {} dynamic-shaped tensors".format(
      len(tensors),
      ', '.join([repr(dt) for dt in set(dtypes)]),
      sum(nonempty_sizes) / 1024.0 / 1024.0,
      len(sizes) - len(nonempty_sizes))

def get_cpu_count():
  """Available cpu count."""
  return int(os.getenv('NUMBER_OF_PROCESSORS', multiprocessing.cpu_count()))


class CoalescingRewriter(Rewriter):
  """Rewriter to split communication to buffers."""

  def __init__(self, max_splits=None, issorted=False, enable_fp16=False,
               fp16_scale=None, enable_logging=False, comm_pool=None):
    """Create a rewriter using buffered communication.

    Args:
      max_splits: (Optional.) max number of splits.
      issorted: (Optional.) True if values already sorted by ticks.
      enable_fp16: (Optional.) True if enabled.
      fp16_scale: (Optional.) Scale the values before sparse allreduce.
      enable_logging: (Optional.) True if logging is enabled.
      comm_pool: (Optional.) Reuse communication pool if comm_pool is not None.
    """
    self._max_splits = max_splits
    if self._max_splits is None:
      self._max_splits = 10
      # By default, it should not use too many threads. For custom max_splits,
      # `inter_op_parallelism_threads` in `tf.ConfigProto` as `session_config`
      # of `tf.estimator.RunConfig` should be increased to prevent hang in
      # local variables and resources initialization.
      # Reserve most 50% of threads for collective ops.
      compute_threads_limit = int(get_cpu_count() * 0.5)
      # At least 1 thread for collective ops.
      if compute_threads_limit < 1:
        compute_threads_limit = 1
      if self._max_splits > compute_threads_limit:
        self._max_splits = compute_threads_limit
    self._enable_fp16 = enable_fp16
    self._fp16_scale = fp16_scale
    self._issorted = issorted
    self.comm_pool = comm_pool
    if not comm_pool:
      raise ValueError("comm_pool should not be None")
    self._enable_logging = enable_logging

  @property
  def max_splits(self):
    """Max number to split."""
    return self._max_splits

  @property
  def issorted(self):
    """True if values are alreadt sorted."""
    return self._issorted

  @property
  def enable_logging(self):
    """True if logging is enabled."""
    return self._enable_logging

  def rewrite(self, device_id, fn, values, comm_name, comm_spec, **kwargs):
    """Call function on specific device.

    Args:
      device_id: device index.
      fn: values, comm_name, comm_spec, **kwargs -> value.
      values: a value or list of values.
      comm_name: unique name for this rewriter.
      comm_spec: communicator builder.
      kwargs: (Optional.) key-value arguments of fn.

    Returns:
      result values.
    """
    tensors = [ops.convert_to_tensor(v) for v in values]
    if device_id == 0 and self.enable_logging:
      logging.info(
          '{} tensors: {}'.format(
              comm_name, tensors_debug_string(tensors)))

    if self._enable_fp16:
      tensors = [self._convert_to_fp16(t, self._fp16_scale) for t in tensors]

    sorted_values, sorted_ticks, sorted_indices = sort_by_tick(
        tensors, issorted=self.issorted)
    # Split sorted values to tuples.
    tuples = split_to_tuples(sorted_values, sorted_ticks, self.max_splits)
    if device_id == 0 and self.enable_logging:
      for tupid, tup in enumerate(tuples):
        logging.info(
            '{} group ({}/{}): {}'.format(
                comm_name, tupid + 1, len(tuples),
                tensors_debug_string(tup)))
    # Flatten tuples.
    tuple_buffer_shape_sizes = []
    for tupid, tup in enumerate(tuples):
      with ops.name_scope("group_{}".format(tupid)):
        tuple_buffer_shape_sizes.append(flatten_to_buffer(tup))
    tuple_buffers, tuple_shapes, tuple_sizes = zip(*tuple_buffer_shape_sizes)
    # Execute on flattened tensors.
    reduced_buffers = []

    reduced_buffers = self.comm_pool.communicate(tuple_buffers)

    # Unflatten result values.
    result_tuples = [
        deflatten_to_tensors(t, tuple_shapes[tupid], tuple_sizes[tupid]) \
        for tupid, t in enumerate(reduced_buffers)]
    # Unsplit tuples.
    sorted_result_tensors = unsplit_to_tensors(result_tuples)
    # Unsort result tensors.
    unsorted_result_tensors = \
      unsort_to_tensors(sorted_result_tensors, sorted_indices)
    if self._enable_fp16:
      result_tensors = [
          self._convert_to_fp32(t, self._fp16_scale) \
          for t in unsorted_result_tensors]
      return result_tensors
    return unsorted_result_tensors
