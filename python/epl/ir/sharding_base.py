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
"""Implementation of ShardingBase."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ShardingBase(object):
  """Base class for a sharding unit. All shardable calculation unit is a
  ShardingBase object."""
  def __init__(self, args, kwargs, name):
    """init ShardingBase.
    Args:
      args: args got from hooked function.
      kwargs: kwargs got from hooked function.
      name: object name.
    """
    self.args = args
    self.kwargs = kwargs
    self.name = name
