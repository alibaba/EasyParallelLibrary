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
"""Schedule strategy of pipeline."""

from epl.parallel import ops


class PipelineScheduler(object):
  """Base class for pipeline scheduler."""
  def __init__(self, num_micro_batch, num_stages):
    """Create pipeline scheduler.

    Args:
      num_micro_batch: Integer value, number of pipeline micro batch.
      num_stages: Number stages of a replica.
    """
    self._num_micro_batch = num_micro_batch
    self._num_stages = num_stages

  def call(self, customs):
    raise NotImplementedError('Must be implemented by subclass.')


class PreferForward(PipelineScheduler):
  """PreferForward scheduler."""
  def call(self, customs):
    for micro_batch_idx in range(1, self._num_micro_batch):
      for stage_idx in range(self._num_stages):
        ops.add_control_dependency(
            customs[stage_idx].forward_exit_ops[micro_batch_idx - 1],
            customs[stage_idx].forward_entrance_ops[micro_batch_idx])
        ops.add_control_dependency(
            customs[stage_idx].backward_exit_ops[micro_batch_idx - 1],
            customs[stage_idx].backward_entrance_ops[micro_batch_idx])
        if micro_batch_idx == self._num_micro_batch - 1:
          ops.add_control_dependency(
              customs[stage_idx].forward_exit_ops[-1],
              customs[stage_idx].backward_entrance_ops[0])


class PreferBackward(PipelineScheduler):
  """PreferBackward scheduler."""
  def call(self, customs):
    for micro_batch_idx in range(1, self._num_micro_batch):
      for stage_idx in range(self._num_stages):
        if stage_idx == self._num_stages - 1:
          ops.add_control_dependency(
              customs[-1].backward_exit_ops[micro_batch_idx - 1],
              customs[-1].forward_entrance_ops[micro_batch_idx])
        else:
          forward_cache_num = self._num_stages
          if micro_batch_idx + stage_idx >= forward_cache_num:
            reverse_idx = stage_idx + micro_batch_idx - forward_cache_num
            if micro_batch_idx == self._num_micro_batch - 1:
              ops.add_control_dependency(
                  customs[stage_idx].forward_exit_ops[micro_batch_idx],
                  customs[stage_idx].backward_entrance_ops[reverse_idx + 1])
            ops.add_control_dependency(
                customs[stage_idx].forward_exit_ops[micro_batch_idx - 1],
                customs[stage_idx].backward_entrance_ops[reverse_idx])
            ops.add_control_dependency(
                customs[stage_idx].backward_exit_ops[reverse_idx],
                customs[stage_idx].forward_entrance_ops[micro_batch_idx])
          else:
            ops.add_control_dependency(
                customs[stage_idx].forward_exit_ops[micro_batch_idx - 1],
                customs[stage_idx].forward_entrance_ops[micro_batch_idx])
            ops.add_control_dependency(
                customs[stage_idx].backward_exit_ops[self._num_micro_batch -
                                                     micro_batch_idx - 1],
                customs[stage_idx].backward_entrance_ops[self._num_micro_batch
                                                         - micro_batch_idx])


class PreferBackwardOptimizer(PipelineScheduler):
  """PreferBackward scheduler."""
  def call(self, customs):
    for micro_batch_idx in range(1, self._num_micro_batch):
      for stage_idx in range(self._num_stages):
        if stage_idx == self._num_stages - 1:
          ops.add_control_dependency(
              customs[-1].backward_exit_ops[micro_batch_idx - 1],
              customs[-1].forward_entrance_ops[micro_batch_idx])
        else:
          forward_cache_num = self._num_stages
          if micro_batch_idx + stage_idx > forward_cache_num:
            reverse_idx = stage_idx + micro_batch_idx - forward_cache_num - 1
            if micro_batch_idx == self._num_micro_batch - 1:
              ops.add_control_dependency(
                  customs[stage_idx].forward_exit_ops[micro_batch_idx],
                  customs[stage_idx].backward_entrance_ops[reverse_idx + 1])
            ops.add_control_dependency(
                customs[stage_idx].forward_exit_ops[micro_batch_idx - 1],
                customs[stage_idx].backward_entrance_ops[reverse_idx])
            ops.add_control_dependency(
                customs[stage_idx].backward_exit_ops[reverse_idx],
                customs[stage_idx].forward_entrance_ops[micro_batch_idx])
          else:
            ops.add_control_dependency(
                customs[stage_idx].forward_exit_ops[micro_batch_idx - 1],
                customs[stage_idx].forward_entrance_ops[micro_batch_idx])
            ops.add_control_dependency(
                customs[stage_idx].backward_exit_ops[self._num_micro_batch - micro_batch_idx - 1],
                customs[stage_idx].backward_entrance_ops[self._num_micro_batch - micro_batch_idx])


# A dictionary to store all scheduler names
SCHEDULER = {
    "preferforward": PreferForward,
    "preferbackward": PreferBackward,
    "preferbackwardoptimizer": PreferBackwardOptimizer,
}

def get_scheduler(name):
  """Get scheduler by name."""
  name = name.lower()
  if name not in SCHEDULER:
    raise RuntimeError("Unknown scheduler {}, current supported schedulers are {}".format(name, SCHEDULER.keys()))
  return SCHEDULER[name]
