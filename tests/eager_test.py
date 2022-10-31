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
"""Eager mode test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.python.platform import test

import epl


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class EagerTest(test.TestCase):
  def test_eager_error(self):
    assert tf.executing_eagerly() is True
    with self.assertRaises(RuntimeError) as ctx:
      epl.init()
    self.assertTrue("Tensorflow eager mode is not supported by EPL now" in str(ctx.exception))


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  tf.enable_eager_execution()
  test.main()
