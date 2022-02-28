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
"""Test for shape and dimension."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test

from epl.ir.shape import Dimension
from epl.ir.shape import ValueDimension
from epl.ir.shape import ValueReduceType
from epl.ir.shape import Shape


# pylint: disable=missing-docstring,unused-variable
class ShapeTest(test.TestCase):
  def test_dimension(self):
    d1 = Dimension(4, 2, 1)
    d2 = Dimension(4, 2, 1)
    d3 = Dimension(4, 2, 0)
    self.assertEqual(d1.value, 4)
    self.assertEqual(d1.sharding_num, 2)
    self.assertEqual(d1.sharding_index, 1)
    self.assertEqual(d1, d2)
    self.assertNotEqual(d2, d3)

    # Test sharding index larger than sharding numebr.
    with self.assertRaises(ValueError):
      d4 = Dimension(4, 1, 2)

    # Test sharding number larger than dimension value.
    with self.assertRaises(ValueError):
      d5 = Dimension(4, 5, 1)

    # Test parameter type.
    with self.assertRaises(ValueError):
      d6 = Dimension(4, '8', '6')

  def test_value_dimension(self):
    d1 = ValueDimension()
    d2 = ValueDimension(reduce_op=ValueReduceType.SUM)
    d3 = ValueDimension(sharding_num=8,
                        sharding_index=7,
                        reduce_op=ValueReduceType.MEAN)

    self.assertNotEqual(1, 4)
    self.assertEqual(d1, d2)
    self.assertNotEqual(d1, d3)
    self.assertEqual(d3.value, 0)
    self.assertEqual(d3.sharding_num, 8)
    self.assertEqual(d3.sharding_index, 7)
    self.assertEqual(d3.reduce_op, ValueReduceType.MEAN)

    # Test reduce type error
    with self.assertRaises(ValueError):
      d4 = ValueDimension(reduce_op=4)

  def test_shape(self):
    # Test basic member values.
    d1 = Dimension(4, 2, 1)
    d2 = Dimension(5, 2, 1)
    d3 = Dimension(4, 2, 0)
    d4 = ValueDimension(sharding_num=5,
                        sharding_index=3,
                        reduce_op=ValueReduceType.MEAN)
    s1 = Shape([d1, d2, d3], d4)
    self.assertEqual(s1.rank, 3)
    self.assertEqual(s1.as_list(), [4, 5, 4])
    self.assertEqual(s1.dims[0], d1)
    self.assertEqual(s1.dims[1], d2)
    self.assertEqual(s1.dims[2], d3)
    self.assertEqual(s1.value_dim, d4)

    # Test equal function.
    s2 = Shape([d1, d2, d3], d4)
    self.assertEqual(s1, s2)

    s3 = Shape([d2, d3])
    self.assertNotEqual(s1, s3)
    self.assertEqual(s3.value_dim, None)


# pylint: enable=missing-docstring,unused-variable

if __name__ == "__main__":
  test.main()
