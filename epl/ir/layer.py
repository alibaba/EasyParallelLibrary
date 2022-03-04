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
"""Implementation of Layer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from epl.ir.sharding_base import ShardingBase


class Layer(ShardingBase):
  """A shardable layer which consists mutilple operations."""

  # TODO(wangang.wa): Here we get call parameters from call function.
  # Actually, we should get parameter from keras layer init function.
  # These parameters could be got from hook_object by traverse its keys
  # in __dict__.
  def __init__(self, hook_object, args, kwargs):
    """Create a new Layer.
    Args:
      hook_object: A tf keras layer object.
      args: keras layer call args.
      kwargs: keras layer call kwargs.
    """
    self.module = hook_object.__module__
    name = hook_object.__class__.__name__

    super(Layer, self).__init__(args, kwargs, name)

  def __str__(self):
    return "Layer(name: %s, module: %s, id: %s)" % (self.name, self.module,
                                                    id(self))
