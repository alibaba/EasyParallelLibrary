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
"""Test for operation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import test
import epl

# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class OperationTest(test.TestCase):
  """Test cases for operation."""

  def test_str(self):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    tf.ones([1, 2], name="test")
    epl_op = epl.Graph.get().operations["test"]
    device = "/job:worker/replica:0/task:0/device:GPU:0"
    phase = "MODEL_FORWARD_PHASE"
    true_value = "epl.Operation(name='test', device={}, ".format(device) + \
        "type=Const, function=None, phase={})".format(phase)
    self.assertEqual(str(epl_op), true_value)


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  test.main()
