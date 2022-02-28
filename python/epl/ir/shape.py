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
"""Meta classes to describe shape information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum

from tensorflow.python.framework import tensor_shape


class Dimension(tensor_shape.Dimension):
  """Tensor dimension with sharding information."""
  def __init__(self, value, sharding_num=0, sharding_index=0):
    """Create a Dimension.

    Inherit from tensorflow Dimension. EPL Dimension has the basic
    abilities to describe the dimension of tensors. It also contains
    sharding information about sharding number and sharding index.

    Args:
      value: dimension value.
      sharding_num: partition number for this dimension.
      sharding_index: partition index for this part.
    """
    super(Dimension, self).__init__(value)
    if not isinstance(sharding_num, int):
      raise ValueError("sharding_num should be a integer number, "
                       "but got: %s." % type(sharding_num))
    if not isinstance(sharding_index, int):
      raise ValueError("sharding_index should be a integer number, "
                       "but got: %s." % type(sharding_index))
    sharding_num = max(sharding_num, 0)
    sharding_index = max(sharding_index, 0)
    if value != 0 and sharding_num > value:
      raise ValueError("sharding_num should be smaller than dimension value, "
                       "but got value: %s, sharding_num: %s ." %
                       (value, sharding_num))
    if sharding_num != 0 and sharding_index >= sharding_num:
      raise ValueError("sharding_index should smaller than shard_num, "
                       "but got sharding_num: %s, sharding_index: %s ." %
                       (sharding_num, sharding_index))
    self.sharding_num = sharding_num
    self.sharding_index = sharding_index

  def __repr__(self):
    return "Dimension(val:%s, num:%s, idx:%s)" % (repr(
        self._value), self.sharding_num, self.sharding_index)

  def __str__(self):
    return self.__repr__()

  def __eq__(self, other):
    if not isinstance(other, Dimension):
      return False
    return super(Dimension, self).__eq__(
        other) and self.sharding_num == other.sharding_num \
               and self.sharding_index == other.sharding_index

  def __ne__(self, other):
    return not self.__eq__(other)


class ValueReduceType(enum.Enum):
  # Sum the values of each parition.
  SUM = 1
  # Mean the values of each partition.
  MEAN = 2
  # Max the values of each parition.
  MAX = 3
  # Min the values in each position.
  MIN = 4


class ValueDimension(Dimension):
  """Type to represent value dimension. The difference between ValueDimension
  and Dimension is that ValueDimension has reduce_op and it has no value.
  ValueDimension is another dimension for tensor in value dim.

  For example, the first input of a matmul is splitted in row dimension
  and the second input of the matmul is splitted in column dimension.
  After sharding the matmul op into 2 parts. One of it output (tesnor-0)
  on worker 0 is
  [[1, 2],
   [3, 4]] .

  A another output tensor (tensor-1) on worker 1 is:
  [[5, 6],
   [7, 8]] .

  To get the final result of the matmul, we should reduce sum the two output
  values together. So the result is:
  [[6, 8],
   [10, 12]].

  Reduce tensors in value dimension to get the final result. So we add
  a value dimension member to the tensors.
  For tensor-0, its ValueDimension contains sharding number=2, sharding
  index=0, reduce_op = SUM.
  For tensor-1, its ValueDimension contains sharding number=2, sharding
  index=0, reduce_op = SUM.
  """
  def __init__(self,
               sharding_num=0,
               sharding_index=0,
               reduce_op=ValueReduceType.SUM):
    """Create a ValueDimension.

    args:
      sharding_num: partition number for this dimension.
      sharding_index: partition index for this part.
      reduce_op: reduce method. Its value should be one of [
        ValueReduceType.SUM,
        ValueReduceType.MEAN,
        ValueReduceType.MAX,
        ValueReduceType.MIN]. Default value is ValueReduceType.SUM.
    """
    super(ValueDimension, self).__init__(0, sharding_num, sharding_index)
    if reduce_op not in [
        ValueReduceType.SUM, ValueReduceType.MEAN, ValueReduceType.MAX,
        ValueReduceType.MIN
    ]:
      raise ValueError("reduce op should be one type of [%s, %s, %s, %s]" %
                       (ValueReduceType.MEAN, ValueReduceType.SUM,
                        ValueReduceType.MAX, ValueReduceType.MIN))
    self._reduce_op = reduce_op

  @property
  def reduce_op(self):
    return self._reduce_op

  def __eq__(self, other):
    if not isinstance(other, ValueDimension):
      return False
    return super(ValueDimension,
                 self).__eq__(other) and self._reduce_op == other.reduce_op

  def __ne__(self, other):
    return not self.__eq__(other)

  def __repr__(self):
    return "ValueDimension(val:%s, num:%s, idx:%s, op:%s)" % (repr(
        self._value), self.sharding_num, self.sharding_index, self.reduce_op)

  def __str__(self):
    return self.__repr__()


def as_dimension(value):
  """Converts the given value to a Dimension.
  Args:
    value: The value to be converted.
  Returns:
    A Dimension corresponding to the given value.
  """
  if isinstance(value, Dimension):
    return value
  return Dimension(value)


tensor_shape.as_dimension = as_dimension


class Shape(tensor_shape.TensorShape):
  """Shape with sharidng information."""
  def __init__(self, dims, value_dim=None):
    """Create a Shape.

    Args:
      dims: A list of Dimensions, or None if the shape is unspecified.
      value_dim: value dimension. Default value is None.
    """
    super(Shape, self).__init__(dims)
    if value_dim and not isinstance(value_dim, ValueDimension):
      raise ValueError("value_dim should be type of ValueDimension, "
                       "but got type: %s." % type(value_dim))
    self.value_dim = value_dim

  def __repr__(self):
    if self.rank is None:
      return "<unknown>"
    return "Shape(Dim:(%s), Val:(%s))" % (", ".join(
        str(d) for d in self._dims), self.value_dim)

  def __str__(self):
    return self.__repr__()

  def __eq__(self, other):
    """Returns True if `self` is equivalent to `other`."""
    return self._dims == other.dims and self.value_dim == other.value_dim

  def __ne__(self, other):
    return not self.__eq__(other)
