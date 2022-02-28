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
"""Implementation base class of parallelism strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect as ins

_STACK_DEPTH = 5
_FILE_NAME_POSITION = 1
_LINE_NUMBER_POSITION = 2


class ParallelStrategy(object):  # pylint: disable=invalid-name
  """Base class for epl strategy which represent different parallelisms."""

  def __init__(self, device_count=None, name=None):
    """Create base strategy.

    Args:
      device_count: number of devices for one TaskGraph replica.
      name: name of the strategy.
    """
    self.devices = None
    self.device_count = device_count
    self.name = name
    # The sequencial index of defined strategy which will be setted
    # by Env context.
    self.index = None
    # Set as default strategy.
    self.is_default = False
    self.stack = self._get_stack()

  def _get_stack(self):
    ins_stack = ins.stack()
    if len(ins_stack) < _STACK_DEPTH:
      raise RuntimeError(
          "Abnormal stack found in strategy constructor, stack: ", ins_stack)
    stack = ""
    for i in range(1, _STACK_DEPTH):
      stack += "%s:%d;" % (ins_stack[i][_FILE_NAME_POSITION],
                           ins_stack[i][_LINE_NUMBER_POSITION])
    return stack

  def __enter__(self):
    from epl.env import Env
    if Env.get().config.auto.auto_parallel:
      raise Exception("auto parallel is enabled,"
                      " do not use any stategies at the same time")
    strategy_context = Env.get().strategy_context
    if strategy_context.default_strategy:
      strategy_context.del_context(strategy_context.default_strategy)
    strategy_context.add_context(self)

  def __exit__(self, unused_exception_type, unused_exception_value,
               unused_traceback):
    from epl.env import Env
    strategy_context = Env.get().strategy_context
    strategy_context.del_context(self)
    if strategy_context.default_strategy:
      strategy_context.add_context(strategy_context.default_strategy)

  def __str__(self):
    return ("strategy('%s', Index=%s, Devices=%s, device_count=%s, is_default=%s)" %
            (self.name, self.index, self.devices, self.device_count, self.is_default))

  def __repr__(self):
    return self.__str__()
