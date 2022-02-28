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
"""Test for Env."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework import ops
import epl
from epl.env import Env


# pylint: disable=missing-docstring,unused-variable
# pylint: disable=protected-access
class RunEnvTest(test.TestCase):
  # Test init and build graph multiple times
  def test_init_multiple(self):
    epl.init()
    with tf.Graph().as_default() as g:
      with epl.Cluster():
        default_graph = ops.get_default_graph()
        default_device = list(default_graph._device_function_stack \
            .peek_traceable_objs())[0].obj.display_name
        default_device = default_device.split('/')[-1]
        self.assertEqual(default_device, 'device:GPU:0')

    epl.init()
    with tf.Graph().as_default() as g:
      with epl.Cluster():
        default_graph = ops.get_default_graph()
        default_device = list(default_graph._device_function_stack \
            .peek_traceable_objs())[0].obj.display_name
        default_device = default_device.split('/')[-1]
        self.assertEqual(default_device, 'device:GPU:0')

  def test_auto_parallel(self):
    epl.init()
    env = Env.get()
    self.assertEqual(env.config.auto.auto_parallel, False)
    env.config.auto.auto_parallel = True
    self.assertEqual(env.config.auto.auto_parallel, True)
    epl.init()
    self.assertEqual(env.config.auto.auto_parallel, False)


# pylint: enable=missing-docstring,unused-variable
# pylint: enable=protected-access

if __name__ == "__main__":
  test.main()
