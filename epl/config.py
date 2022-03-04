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
"""Classes for epl configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import os
from epl.utils import constant


class BaseConfig(object):
  """Base class includes some common format functions."""

  def __init__(self):
    self._finalize = True

  def __str__(self):
    members = [attr for attr in dir(self) \
        if not callable(getattr(self, attr)) and not attr.startswith("__")]
    ser_str = self.__class__.__name__ + " {\n"
    for key in members:
      if key.startswith('_'):
        continue
      attr = getattr(self, key)
      attr = '"{}"'.format(attr) if isinstance(attr, str) else attr
      ser_str += "    %s = %s,\n" % (key, attr)
    ser_str += "}"

    return ser_str

  def __repr__(self):
    return self.__str__()

  def __setattr__(self, name, value):
    """Avoid adding new attributes by users."""
    if name != "_finalize" and self._finalize and not hasattr(self, name):
      raise AttributeError('{} instance has no attribute {!r}'.format(type(self).__name__, name))
    super(BaseConfig, self).__setattr__(name, value)

class AutoParallelConfig(BaseConfig):
  """Config for auto parallel."""

  # Whether auto parallel is enabled.
  auto_parallel = False


class IOConfig(BaseConfig):
  """Config for io."""

  # Drop last files to balance dataset files between workers.
  drop_last_files = False

  # Allowed unbalanced io slicing if unbalanced_io_slicing is True.
  unbalanced_io_slicing = False

  # IO slicing switch. Default value is False. Set io_slicing with
  # True could enable auto io slicing mechanism. Remeber to remove
  # your own io slicing code when io_slicing is True.
  slicing = False


class CommunicationConfig(BaseConfig):
  """Config for communication."""

  # Convert sparse gradients to dense before communication.
  sparse_as_dense = False

  # The max group number for gradients fusion.
  max_splits = 5

  # The number of communicators in communication pool. Default value is
  # 2 means will create a communication pool with 2 communicators.
  num_communicators = 2

  # Compress tensors to float 16 before communication.
  fp16 = False

  # Scale value when enable tensors compression with float 16.
  fp16_scale = 128

  # Excute clip_by_norm after gradient aggregation. Default value is False.
  clip_after_allreduce = False

  # Gradients allreduce method. Default values is mean.
  gradients_reduce_method = constant.REDUCE_METHOD_MEAN


class PipelineConfig(BaseConfig):
  """Config for pipeline."""

  # Stage num in auto mode.
  num_stages = -1

  # Pipeline num_micro_batch, default is set to 1.
  num_micro_batch = 1

  # Pipeline strategy.
  strategy = constant.DEFAUT_PIPELINE_STRATEGY


class GradientCheckpointConfig(BaseConfig):
  """Config for gradient checkpoint."""

  # Gradient checkpoints type: collection, auto. default is empty.
  type = ""

  # Last taskgraph to apply GC, used in auto GC.
  end_taskgraph = -1

  # Check GC gradients with base gradients.
  check_gradients = False


class ZeroConfig(BaseConfig):
  """Config for ZERO."""

  # Enable zero by set zero_level to v0, v1, v2.
  # v0 partitions the optimizer states.
  # v1 partitions optimizer states and gradients.
  # v2 partitions weights, gradients and optimizer states.
  # Now v0 and v1 are supported.
  level = ""


class OffloadConfig(BaseConfig):
  """Config for offload."""

  # offload level
  # v0 means offload all variables
  level = ""


class AMPConfig(BaseConfig):
  """Config for amp."""

  # auto mixed precision level, currently only support O1.
  level = ""

  # Enable amp debug log.
  debug_log = False

  # Loss scale for amp, can be "dynamic" or number(for fix).
  loss_scale = "dynamic"


class ClusterConfig(BaseConfig):
  """Config for cluster."""

  # Prefer placing one model replica within node.
  device_place_prefer_intra_node = True

  # Visible devices for session. Usually, its value is setted by scheduler.
  run_visible_devices = ""

  # Place split and replicate taskgraphs into the same device.
  colocate_split_and_replicate = False


class OptimizerConfig(BaseConfig):
  """Config for optimizer."""

  # Number of gradient apply groups.
  num_apply_group = 1


class Config(BaseConfig):
  """A class to manage epl configuration.

  Args:
    param_dict: Dict format of parameters."""

  def __init__(self, param_dict=None):
    super(Config, self).__init__()
    self._finalize = False
    # Auto parallel config.
    self.auto = AutoParallelConfig()
    # IO config.
    self.io = IOConfig()
    # Communication config.
    self.communication = CommunicationConfig()
    # Pipeline config.
    self.pipeline = PipelineConfig()
    # Gradient Checkpoint config.
    self.gradient_checkpoint = GradientCheckpointConfig()
    # ZERO config.
    self.zero = ZeroConfig()
    # Offload config.
    self.offload = OffloadConfig()
    # AMP config.
    self.amp = AMPConfig()
    # Cluster config.
    self.cluster = ClusterConfig()
    # optimizer config.
    self.optimizer = OptimizerConfig()

    self._parse_params(param_dict)
    self._finalize = True
    self._validate_params()

  def _parse_params(self, param_dict):
    """Parse params from param_dict or ENV."""
    if not param_dict:
      param_dict = {}

    def get_string_from_env(key, default_value):
      """Get string parameter from environ."""
      return os.environ.get(key, default_value)

    def get_int_from_env(key, default_value):
      """Get int parameter from environ."""
      value = os.environ.get(key, default_value)
      if value is None and default_value is None:
        return None
      return int(value)

    def get_float_from_env(key, default_value):
      """Get float parameter from environ."""
      value = os.environ.get(key, default_value)
      if value is None and default_value is None:
        return None
      return float(value)

    def get_bool_from_env(key, default_value):
      """Get bool parameter from environ."""
      value = os.environ.get(key)
      if value is None:
        return default_value
      elif value.lower() == "true":
        return True
      elif value.lower() == "false":
        return False
      else:
        raise ValueError("Unknown bool parameter, key: %s, value: %s" % (key, value))

    def _get_attributes(cls):
      """Get attributes from class."""
      return [(name, attr) for name, attr in inspect.getmembers(cls) if not name.startswith('_')]

    def _get_and_check_type(value, default_value, key):
      # To be noticed: all str type values should in lower case.
      if isinstance(value, str):
        value = value.lower()
      if default_value is None:
        return value
      if not isinstance(value, type(default_value)):
        raise ValueError("%s type error, expected: %s." \
                          % (key, type(default_value)))
      return value

    def _get_env_func(default_value, default_env_func=None):
      """Return the function of get params from env."""
      if default_env_func:
        return default_env_func
      if isinstance(default_value, bool):
        return get_bool_from_env
      if isinstance(default_value, int):
        return get_int_from_env
      if isinstance(default_value, str):
        return get_string_from_env
      if isinstance(default_value, float):
        return get_float_from_env
      raise RuntimeError("%s type is not supported for now." % type(default_value))

    for name, conf in _get_attributes(self):
      for sub_name, default_value in _get_attributes(conf):
        actual_value = default_value

        # Set config from env.
        # e.g. EPL_PIPELINE_NUM_MICRO_BATCH
        env_config_name = ('epl_' + name + '_' + sub_name).upper()
        if env_config_name in os.environ:
          actual_value = _get_env_func(default_value)(env_config_name, actual_value)

        # Set config from python code, overwrite env config if both are set.
        # e.g. pipeline.num_micro_batch
        config_name = name + '.' + sub_name
        if config_name in param_dict:
          actual_value = param_dict[config_name]
        if config_name == "amp.loss_scale":
          if actual_value != "dynamic":
            actual_value = float(actual_value)
        else:
          actual_value = _get_and_check_type(actual_value, default_value, config_name)
        setattr(conf, sub_name, actual_value)

  def _validate_params(self):
    reduce_methods = [constant.REDUCE_METHOD_MEAN, constant.REDUCE_METHOD_SUM]
    if self.communication.gradients_reduce_method not in reduce_methods:
      raise ValueError("Gradients reduce method error: %s, which should be one of %s." %
                       (self.communication.gradients_reduce_method, reduce_methods))
