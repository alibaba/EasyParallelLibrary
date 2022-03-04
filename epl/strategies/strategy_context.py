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
"""Implementation of strategy context."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from epl.strategies.parallel_strategy import ParallelStrategy
from epl.strategies.replicate import Replicate
from epl.strategies.split import Split


class StrategyContext(object):
  """Manage the strategy context of epl."""
  def __init__(self):
    self._state = []
    self._strategy_count_each_type = {}
    self.update_flag = True
    self._default_strategy = None

  def add_context_check(self, strategy):
    """Validation check of new added strategy."""

    #TODO(wangang.wa): Should consider more conditions.

    # not support nest strategies with same type
    if isinstance(strategy, tuple([type(t) for t in self._state])):
      raise RuntimeError("Can't nest strategies with same type. Current strategy"
                         ": {}, Context: {} .".format(strategy, self._state))

    # not support nest any strategy in split
    if Split in [type(t) for t in self._state]:
      raise RuntimeError("Can't nest strategies in split strategy. Current strategy"
                         ": {}, Context: {} .".format(strategy, self._state))

    # not support nest split in replicate
    if isinstance(strategy, Split) and self.replicate_strategy is not None:
      raise RuntimeError("Can't nest split strategy in replicate strategy. Current "
                         "strategy: {}, Context: {} .".format(strategy, self._state))

    return True

  def del_context_check(self, strategy):
    """Validation check of deleted strategy."""

    if len(self._state) < 1:
      return False
    if self._state[-1] is not strategy:
      raise RuntimeError("Remove strategy from epl context failed. "
                         "Current strategy to be removed is not the last one. "
                         "Current strategy: '{}', Context: {} .".format(
                             strategy.name, self._state))
    return True

  def add_context(self, strategy):
    """Add new strategy to context."""
    from epl.env import Env

    if not isinstance(strategy, ParallelStrategy):
      raise ValueError("Strategy type only for add_context parameter. Current "
                       "parameter type: %s." % type(strategy))
    if self.add_context_check(strategy):
      # For split mode, only slice models across all devices is allowed.
      # TODO(jiangle.jl): Supoort nested mode of split strategy and other types of strategy
      if isinstance(strategy, Split) and Env.get().cluster.virtual_devices:
        strategy.devices = Env.get().cluster.virtual_devices[0].all_devices

      if not strategy.is_default:
        if not isinstance(strategy, tuple(self._strategy_count_each_type.keys())):
          self._strategy_count_each_type[type(strategy)] = 1
        strategy.index = sum(self._strategy_count_each_type.values()) - \
                      len(list(self._strategy_count_each_type.values()))

        self._strategy_count_each_type[type(strategy)] += 1
        self.update_flag = True
      self._state.append(strategy)

  def del_context(self, strategy):
    """Delete strategy from context."""

    if self.del_context_check(strategy):
      self._state.remove(strategy)

  @property
  def default_strategy(self):
    return self._default_strategy

  def __str__(self):
    return str(self._state)

  def __repr__(self):
    return self.__str__()

  def __nonzero__(self):
    return bool(self._state)

  # For python 3.x
  __bool__ = __nonzero__

  @property
  def state(self):
    return self._state

  def get_strategy(self, strategy_type):
    """Return strategy object which type is specified strategy type. None will be
    returned if it not exist."""
    for state in self._state:
      if isinstance(state, strategy_type):
        return state
    return None

  @property
  def replicate_strategy(self):
    return self.get_strategy(Replicate)

  @property
  def split_strategy(self):
    return self.get_strategy(Split)

  @property
  def identity(self):
    return hash(str([s.stack for s in self._state]))

  def _reset_default_strategy(self):
    """reset default strategy"""
    if self._default_strategy:
      self._state.remove(self._default_strategy)
      self._default_strategy.is_default = False
      self._default_strategy = None

  @default_strategy.setter
  def default_strategy(self, strategy):
    """Set default strategy."""
    self._reset_default_strategy()
    strategy.is_default = True
    if strategy not in self._state:
      self.add_context(strategy)
      self.update_flag = True
    self._default_strategy = strategy
