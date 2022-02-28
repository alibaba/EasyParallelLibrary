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
"""SummaryInfo class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class SummaryInfo(object):
  """Record tags, tensor.name, symmary_type for some summary."""
  def __init__(self, tags, tensor_name, summary_type):
    self._tags = tags
    self._tensor_name = tensor_name
    self._summary_type = summary_type

  @property
  def tags(self):
    return self._tags

  @property
  def tensor_name(self):
    return self._tensor_name

  @property
  def summary_type(self):
    return self._summary_type

  def serialize(self):
    return "epl.SummaryInfo(tags = {}, tensor_name = {}, summary_type = {}"\
        .format(self.tags, self.tensor_name, self.summary_type)

  def __str__(self):
    return self.serialize()

  def __repr__(self):
    return self.serialize()
