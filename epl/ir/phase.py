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
"""Classes for ModelPhase."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelPhase(object):
  """Phase information to indicate different moment in model defination."""


  ADD_FUNCTION = "ADD_FUNCTION_PHASE"
  APPLY = "MODEL_APPLY_PHASE"
  BACKWARD = "MODEL_BACKWARD_PHASE"
  FORWARD = "MODEL_FORWARD_PHASE"
  FORWARD_AND_BACKWARD_AND_APPLY = "FORWARD_AND_BACKWARD_AND_APPLY_PHASE"
  MICRO_BATCH_CLONE = "MICRO_BATCH_CLONE_PHASE"
  REPLICATED = "REPLICATED_PHASE"
  SAVE_AND_RESTORE = "SAVE_AND_RESTORE_PHASE"
  SESSION_RUN_PHASE = "SESSION_RUN_PHASE"

  def __init__(self, phase):
    """Create a ModelPhase.

    Args:
      phase: Type of ModelPhase static variable represents phase
        of model defination.
    """
    self._phase_backup = None
    self._phase = phase

  def __enter__(self):
    from epl.ir.graph import Graph
    self._phase_backup = Graph.get().current_model_phase
    Graph.get().set_model_phase(self._phase)

  def __exit__(self, unused_exception_type, unused_exception_value,
               unused_traceback):
    from epl.ir.graph import Graph
    Graph.get().set_model_phase(self._phase_backup)
