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
"""Test for distributed initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test
from tensorflow.python.ops import init_ops
from epl.ops import initializers


# pylint: disable=missing-docstring
class InitializersTest(test.TestCase):
  def test_distributed_glorot_uniform(self):
    mode = "fan_in"
    seed = 7
    scale = 8.9
    distribution = "truncated_normal"
    dtype = dtypes.float16
    fan_in = 9
    fan_out = 10

    glorot_uniform = init_ops.GlorotUniform(seed=seed)
    glorot_uniform.mode = mode
    glorot_uniform.scale = scale
    glorot_uniform.distribution = distribution
    glorot_uniform.dtype = dtype
    dist_glorot_uniform = initializers.get(glorot_uniform,
                                           fan_in=fan_in,
                                           fan_out=fan_out)

    self.assertEqual(dist_glorot_uniform.mode, mode)
    self.assertEqual(dist_glorot_uniform.scale, scale)
    self.assertEqual(dist_glorot_uniform.distribution, distribution)
    self.assertEqual(dist_glorot_uniform.dtype, dtype)
    self.assertEqual(dist_glorot_uniform.fan_in, fan_in)
    self.assertEqual(dist_glorot_uniform.fan_out, fan_out)


# pylint: enable=missing-docstring

if __name__ == '__main__':
  test.main()
