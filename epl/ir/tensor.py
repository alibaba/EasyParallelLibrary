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
"""Implementation of graph tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from epl.utils import common


class Tensor(object):
  """A tensor is produced by operation in compuatation graph."""
  def __init__(self, primitive_obj, producer):
    self._primitive_obj = primitive_obj
    self._producer = producer
    self.init()

  def init(self):
    """Tensor initialization."""
    # TODO(wangang.ang): filter duplicated consumers.
    self._consumers = list()
    # Inference shape for cloned tensor from original tensor.
    if common.get_replica_index_from_node_name(self.name) or \
        common.get_micro_batch_index_from_node_name(self.name):
      original_tensor = \
          self.producer.graph.get_tensor_by_name(
              common.get_original_name_from_cloned_object(self.name))
      if not original_tensor:
        return
      self.set_shape(original_tensor.shape)

  @property
  def name(self):
    return self._primitive_obj.name

  @property
  def dtype(self):
    return self._primitive_obj.dtype

  @property
  def device(self):
    return self._primitive_obj.device

  @property
  def producer(self):
    return self._producer

  @property
  def op(self):
    return self._producer

  @property
  def taskgraph(self):
    return self._producer.taskgraph

  def add_consumer(self, consumer):
    if consumer not in self._consumers:
      self._consumers.append(consumer)

  @property
  def consumers(self):
    return self._consumers

  @property
  def primitive_obj(self):
    return self._primitive_obj

  @property
  def shape(self):
    return self._primitive_obj.shape

  def set_shape(self, shape):
    self._primitive_obj.set_shape(shape)

  def __str__(self):
    return "Tensor('%s', shape=%s, dtype=%s, device='%s', phase=%s)" % (
        self._primitive_obj.name, self._primitive_obj.shape,
        self._primitive_obj.dtype, self._primitive_obj.device,
        self.producer.phase)

  def __repr__(self):
    return self.__str__()
