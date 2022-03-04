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
"""Profiler to build cost model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.profiler import ProfileOptionBuilder, profile
from tensorflow.python.framework import ops
from epl.utils.shape_inference import infer_shape, get_tf_tensor_shape


def _get_flops(node, op2flops):
  """Parse profile flops"""
  for children in node.children:
    op2flops[children.name] = int(children.float_ops)
    _get_flops(children, op2flops)
  return op2flops


def profile_flops(is_shape_unknown=True):
  """Profile graph and get flops for each op."""
  opt = ProfileOptionBuilder.float_operation()
  opt = ProfileOptionBuilder(opt).with_empty_output().build()
  g = ops.get_default_graph()
  if is_shape_unknown:
    profile_graph = infer_shape(g)
  flops = profile(profile_graph, cmd='graph', options=opt)
  op2flops = {}
  return _get_flops(flops, op2flops)


def profile_memory(is_shape_unknown=True):
  """Profile graph and get memory for each op"""
  if is_shape_unknown:
    profile_graph = infer_shape(ops.get_default_graph())
  operations = profile_graph.get_operations()
  tensor2bytes = {}
  for op in operations:
    for tensor in op.outputs:
      shape = get_tf_tensor_shape(tensor)
      if shape:
        tbytes = np.prod(shape) * 4
        tensor2bytes[tensor.name] = tbytes
  return tensor2bytes
