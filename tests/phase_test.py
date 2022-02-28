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
"""Test for ModelPhase."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test

import epl
from epl.ir.graph import Graph
from epl.ir.phase import ModelPhase


# pylint: disable=missing-docstring
class PhaseTest(test.TestCase):
  def test_phase(self):
    epl.init()
    g = Graph.get(may_create=True)
    self.assertEqual(g.current_model_phase, ModelPhase.FORWARD)
    g.set_model_phase(ModelPhase.MICRO_BATCH_CLONE)
    with ModelPhase(ModelPhase.BACKWARD):
      self.assertEqual(g.current_model_phase, ModelPhase.BACKWARD)
    self.assertEqual(g.current_model_phase, ModelPhase.MICRO_BATCH_CLONE)


# pylint: enable=missing-docstring

if __name__ == '__main__':
  test.main()
