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
"""Implementation of parallelism strategy of split."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from epl.strategies.parallel_strategy import ParallelStrategy


class Split(ParallelStrategy):  # pylint: disable=invalid-name
  """Classs for model parallelism. Taskgraph in split strategy will be split
  into multiple part in distributed mode."""
  def __init__(self, device_count=None, name=None):
    """Create split strategy.

    Args:
      device_count: number of devices for one TaskGraph replica.
      name: name of the strategy.
    """

    # Disable nested op replacement for a layer or an op
    self._is_nested = False

    super(Split, self).__init__(device_count, 'Split' if name is None else name)

  @property
  def is_nested(self):
    return self._is_nested

  @is_nested.setter
  def is_nested(self, state):
    self._is_nested = state


def split(device_count=None, name=None):
  """epl.split strategy."""
  return Split(device_count, name)
