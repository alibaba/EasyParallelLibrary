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
"""Implementation of distributed initializer."""

import math
import six

from tensorflow.python.framework import dtypes
from tensorflow.python.keras import initializers
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import init_ops


class DistributedGlorotUniform(init_ops.VarianceScaling):
  """The Glorot uniform initializer, also called Xavier uniform initializer.

  It draws samples from a uniform distribution within [-limit, limit]
  where `limit` is `sqrt(6 / (fan_in + fan_out))`
  where `fan_in` is the number of input units in the weight tensor
  and `fan_out` is the number of output units in the weight tensor.
  Args:
    scale: Scaling factor (positive float).
    fan_in: Integer value of fan in.
    fan_out: Integer value of fan out.
    seed: A Python integer. Used to create random seeds. See
        `tf.compat.v1.set_random_seed` for behavior.
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of "normal", "uniform".
    dtype: The data type. Only floating point types are supported.
  References:
    [Glorot et al., 2010](http://proceedings.mlr.press/v9/glorot10a.html)
    ([pdf](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf))
  """
  def __init__(self,
               scale=1.0,
               fan_in=1,
               fan_out=1,
               seed=None,
               mode="fan_avg",
               distribution="uniform",
               dtype=dtypes.float32):
    self.fan_in = fan_in
    self.fan_out = fan_out
    super(DistributedGlorotUniform, self).__init__(scale=scale,
                                                   mode=mode,
                                                   distribution=distribution,
                                                   seed=seed,
                                                   dtype=dtype)

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    scale /= max(1., (self.fan_in + self.fan_out) / 2.)
    limit = math.sqrt(3.0 * scale)
    return random_ops.random_uniform(shape,
                                     -limit,
                                     limit,
                                     dtype,
                                     seed=self.seed)

  def get_config(self):
    return {
        "fan_in": self.fan_in,
        "fan_out": self.fan_out,
        "seed": self.seed,
        "dtype": self.dtype.name
    }


def get(identifier, **kwargs):
  """Get initializer by indentify information."""
  if identifier is None:
    return DistributedGlorotUniform(fan_in=kwargs.get("fan_in", 1),
                                    fan_out=kwargs.get("fan_out", 1))
  elif isinstance(identifier, init_ops.GlorotUniform):
    return DistributedGlorotUniform(scale=identifier.scale,
                                    fan_in=kwargs.get("fan_in", 1),
                                    fan_out=kwargs.get("fan_out", 1),
                                    seed=identifier.seed,
                                    mode=identifier.mode,
                                    distribution=identifier.distribution,
                                    dtype=identifier.dtype)
  elif isinstance(identifier, six.string_types):
    identifier = str(identifier)
    # TODO(wangang.wa): Some other initializers need a distributed
    # implementation if its values related with layer's shape.
    special_cases = ['glorot_uniform']
    if identifier in special_cases:
      return DistributedGlorotUniform(fan_in=kwargs.get("fan_in", 1),
                                      fan_out=kwargs.get("fan_out", 1))
  return initializers.get(identifier)
