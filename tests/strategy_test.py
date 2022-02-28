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
"""Test for strategies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test

import epl
from epl.env import Env


# pylint: disable=missing-docstring,protected-access
class StrategyTest(test.TestCase):
  def test_strategies(self):
    config = epl.Config()
    config.pipeline.num_micro_batch = 2
    epl.init(config)
    stage_strategy_1 = epl.replicate(device_count=1, name='stage_strategy_1')
    stage_strategy_2 = epl.replicate(device_count=1, name='stage_strategy_2')
    context = Env.get().strategy_context
    self.assertEqual(len(context._state), 0)
    with stage_strategy_1:
      self.assertEqual(len(context._state), 1)
      self.assertEqual(context._state[0], stage_strategy_1)
      self.assertEqual(stage_strategy_1.index, 0)
    with stage_strategy_2:
      self.assertEqual(len(context._state), 1)
      self.assertEqual(context._state[0], stage_strategy_2)
      self.assertEqual(stage_strategy_2.index, 1)

  def test_nest_same_type_error(self):
    epl.init()
    with self.assertRaises(RuntimeError):
      with epl.split():
        with epl.split():
          self.fail('Test failed, Runtime error expected.')

    with self.assertRaises(RuntimeError):
      with epl.replicate(name='rep_1'):
        with epl.replicate(name='rep_2'):
          self.fail('Test failed, Runtime error expected.')

    with self.assertRaises(RuntimeError):
      with epl.replicate(device_count=1, name='stage'):
        with epl.replicate(device_count=1, name='stage'):
          self.fail('Test failed, Runtime error expected.')

  def test_nest_stage_error(self):
    epl.init()
    with self.assertRaises(RuntimeError):
      with epl.replicate(device_count=1):
        with epl.split():
          self.fail('Test failed, Runtime error expected.')
    with self.assertRaises(RuntimeError):
      with epl.replicate(device_count=1):
        with epl.replicate():
          self.fail('Test failed, Runtime error expected.')

  def test_nest_split_replica_error(self):
    epl.init()
    with self.assertRaises(RuntimeError):
      with epl.replicate():
        with epl.split():
          self.fail('Test failed, Runtime error expected.')

  def test_strategy_stack(self):
    with epl.replicate():
      self.assertEqual(len(Env.get().strategy_context.state), 1)
      stack = Env.get().strategy_context.state[0].stack
      stack_list = stack.split(";")
      self.assertTrue(len(stack_list) > 3)
      self.assertTrue(stack_list[0].split(":")[0].endswith(
          "parallel_strategy.py"))
      self.assertTrue(stack_list[1].split(":")[0].endswith("replicate.py"))
      self.assertTrue(stack_list[3].split(":")[0].endswith("strategy_test.py"))

  def test_default_strategy(self):
    epl.init()
    r1 = epl.replicate(1)
    r2 = epl.replicate(1)
    epl.set_default_strategy(r1)
    self.assertEqual(len(epl.Graph.get().taskgraphs), 1)
    self.assertEqual(len(Env.get().strategy_context.state), 1)
    self.assertTrue(Env.get().strategy_context.state[0] is r1)
    epl.set_default_strategy(r2)
    self.assertEqual(len(Env.get().strategy_context.state), 1)
    self.assertTrue(Env.get().strategy_context.state[0] is r2)
    self.assertEqual(len(epl.Graph.get().taskgraphs), 2)
    with r1:
      self.assertEqual(len(Env.get().strategy_context.state), 1)
      self.assertTrue(Env.get().strategy_context.state[0] is r1)
      self.assertEqual(len(epl.Graph.get().taskgraphs), 2)
    with r2:
      self.assertEqual(len(Env.get().strategy_context.state), 1)
      self.assertTrue(Env.get().strategy_context.state[0] is r2)
      self.assertEqual(len(epl.Graph.get().taskgraphs), 2)


# pylint: enable=missing-docstring,protected-access

if __name__ == '__main__':
  test.main()
