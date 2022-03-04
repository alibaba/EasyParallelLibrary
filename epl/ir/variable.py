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
"""Implementation of graph variable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from epl.ir.sharding_base import ShardingBase


class Variable(ShardingBase):
  """A logic variable conception which consists multiple operations
  in tensorflow graph"""
  def __init__(self, hook_object, args, kwargs):  # pylint: disable=unused-argument
    """Create a new Variable.
    Args:
      hook_object: A tf variable object. Hook tf RefVariable __init__ function
        to group operations related to this variable together. Be carefull
        when using this parameter for initilizing not ready.
      args: tf variable init args.
      kwargs: tf variable init kwargs.
    """
    name = kwargs.get("name")
    super(Variable, self).__init__(args, kwargs, name)

  def __str__(self):
    return "Variable(name: %s, id: %s)" % (self.name, id(self))
