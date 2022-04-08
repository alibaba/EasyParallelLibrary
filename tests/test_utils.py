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
"""Test utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import numpy as np
import tensorflow as tf

def fix_randomness():
  """Fix model randomness."""
  seed = 123123
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  os.environ['PYTHONHASHSEED'] = str(seed)
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)

def input_to_tensorarray(value, axis, size=None):
  """create input to tensorarray for while_loop."""
  shape = value.get_shape().as_list()
  rank = len(shape)
  dtype = value.dtype
  array_size = shape[axis] if not shape[axis] is None else size

  if array_size is None:
    raise ValueError("Can't create TensorArray with size None")

  array = tf.TensorArray(dtype=dtype, size=array_size)
  dim_permutation = [axis] + list(range(1, axis)) + [0] + \
      list(range(axis + 1, rank))
  unpack_axis_major_value = tf.transpose(value, dim_permutation)
  full_array = array.unstack(unpack_axis_major_value)
  return full_array
