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
"""Rewriters for distributed training optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

from epl.communicators.base import Communicator

class Rewriter(object): # pylint: disable=useless-object-inheritance
  """Base class for all rewriters.

  Rewriter is a python decorator to wrap a function with signature
  `value, comm=communicator_instance, **kwargs -> value` for distribution
  training optimization.
  """

  def decorated_call(self, fn, values, comm_name, comm_spec, **kwargs):
    """Call function for a value or list of values.

    Args:
      fn: value, comm=communicator_instance, **kwargs -> value.
      values: a value or list of values.
      kwargs: (Optional.) key-value arguments of fn.

    Returns:
      result values.
    """
    with ops.name_scope(comm_name):
      current_device = Communicator.current_device()
      # Fast path for single value.
      if not isinstance(values, (tuple, list)):
        default_rank = comm_spec.devices.index(current_device)
        return self.rewrite(
            default_rank, fn, [values], comm_name, comm_spec, **kwargs)[0]

      if not values:
        return []

      default_rank = comm_spec.devices.index(current_device)
      return self.rewrite(
          default_rank, fn, values, comm_name, comm_spec, **kwargs)

  def __call__(self, fn):
    def decorated(values, comm_name, comm_spec, **kwargs):
      return self.decorated_call(
          fn, values, comm_name, comm_spec, **kwargs)
    return decorated

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
    raise NotImplementedError

  def _convert_to_fp16(self, tensor, scale=None):
    if tensor.dtype.is_floating:
      if scale and scale > 0 and scale != 1:
        return math_ops.cast(math_ops.scalar_mul(scale, tensor),
                             dtypes.float16)
      return math_ops.cast(tensor, dtypes.float16)
    return tensor

  def _convert_to_fp32(self, tensor, scale=None):
    if tensor.dtype.is_floating:
      if scale and scale > 0 and scale != 1:
        return math_ops.scalar_mul(1.0/scale,
                                   math_ops.cast(tensor, dtypes.float32))
      return math_ops.cast(tensor, dtypes.float32)
    return tensor
