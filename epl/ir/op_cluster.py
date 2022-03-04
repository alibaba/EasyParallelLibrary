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
"""Implementation of OpCluster."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from epl.ir.sharding_base import ShardingBase


class OpCluster(ShardingBase):
  """A cluster of operations. Group operation into cluster based on
  function in nn_ops.py, math_ops.py, etc. """
  def __init__(self, hook_object, args, kwargs):
    """Create a new OpCluster.
    Args:
      hook_object: A hooked function. Unsually, it's a function
        from math_ops.py, nn_ops.py, etc. OpCluster gets some cluster
        information from hook_object.
      args: function args.
      kwargs: function kwargs.
    """
    if hasattr(hook_object, "__name__"):
      name = hook_object.__name__
    else:
      name = str(hook_object)
    super(OpCluster, self).__init__(args, kwargs, name)

  def __str__(self):
    return "OpCluster(name: %s, id: %s)" % (self.name, id(self))
