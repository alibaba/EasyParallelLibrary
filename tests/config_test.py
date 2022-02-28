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

import os

from tensorflow.python.platform import test
from epl import init
from epl.cluster import Cluster
from epl.config import Config
from epl.env import Env
from epl.utils.common import get_device_string
from epl.utils import constant

# pylint: disable=missing-docstring,unused-argument
_GPU_PER_WORKER = 8
def _mock_available_gpus():
  def available_gpus(self, *args, **kwargs):
    devices = []
    for gpu_index in range(_GPU_PER_WORKER):
      devices.append(get_device_string(task=0, device_index=gpu_index))
    return devices

  return available_gpus

Cluster.available_gpus = _mock_available_gpus()


class RunConfigTest(test.TestCase):
  """Test for epl config."""
  def test_config_default_value(self):
    """Test default value."""
    conf = Config()
    self.assertEqual(conf.communication.sparse_as_dense, False)
    self.assertEqual(conf.communication.max_splits, 5)
    self.assertEqual(conf.communication.fp16, False)
    self.assertEqual(conf.communication.fp16_scale, 128)
    self.assertEqual(conf.communication.num_communicators, 2)
    self.assertEqual(conf.io.drop_last_files, False)
    self.assertEqual(conf.io.unbalanced_io_slicing, False)
    self.assertEqual(conf.io.slicing, False)
    self.assertEqual(conf.cluster.run_visible_devices, "")
    self.assertEqual(conf.communication.gradients_reduce_method,
                     constant.REDUCE_METHOD_MEAN)

  def test_config_with_specified_value(self):
    """Test specified value by user."""
    conf = Config()
    conf.communication.sparse_as_dense = True
    conf.communication.fp16_scale = 128
    conf.io.drop_last_files = True
    conf.cluster.run_visible_devices = "0,1,2"
    conf.io.slicing = False

    self.assertEqual(conf.communication.sparse_as_dense, True)
    self.assertEqual(conf.communication.fp16_scale, 128)
    self.assertEqual(conf.communication.max_splits, 5)
    self.assertEqual(conf.io.drop_last_files, True)
    self.assertEqual(conf.io.unbalanced_io_slicing, False)
    self.assertEqual(conf.io.slicing, False)
    self.assertEqual(conf.cluster.run_visible_devices, "0,1,2")

  def test_epl_init_with_config(self):
    """Test config value in epl Env."""
    conf = Config({
        "communication.max_splits": 15,
        "communication.fp16_scale": 256,
        "communication.sparse_as_dense": True,
        "io.unbalanced_io_slicing": True,
        "io.drop_last_files": False,
        "io.slicing": True,
        "communication.gradients_reduce_method": 'MEAN'
    })
    init(conf)
    real_conf = Env.get().config
    self.assertEqual(real_conf.communication.max_splits, 15)
    self.assertEqual(real_conf.communication.fp16_scale, 256)
    self.assertEqual(real_conf.communication.sparse_as_dense, True)
    self.assertEqual(real_conf.io.unbalanced_io_slicing, True)
    self.assertEqual(real_conf.io.drop_last_files, False)
    self.assertEqual(real_conf.io.slicing, True)
    self.assertEqual(real_conf.communication.gradients_reduce_method,
                     constant.REDUCE_METHOD_MEAN)

  def test_epl_init_with_env(self):
    """Test config value init from Env and config."""
    os.environ["EPL_COMMUNICATION_SPARSE_AS_DENSE"] = "True"
    os.environ["EPL_COMMUNICATION_MAX_SPLITS"] = "24"
    os.environ["EPL_COMMUNICATION_FP16"] = "True"
    os.environ["EPL_COMMUNICATION_FP16_SCALE"] = "256"
    os.environ["EPL_COMMUNICATION_NUM_COMMUNICATORS"] = "19"
    os.environ["EPL_IO_DROP_LAST_FILES"] = "True"
    os.environ["EPL_IO_SLICING"] = "True"
    os.environ["EPL_CLUSTER_RUN_VISIBLE_DEVICES"] = "0,4,7"
    os.environ["EPL_COMMUNICATION_GRADIENTS_REDUCE_METHOD"] = "SUM"

    conf = Config()
    init(conf)
    real_conf = Env.get().config
    self.assertEqual(real_conf.communication.sparse_as_dense, True)
    self.assertEqual(real_conf.communication.max_splits, 24)
    self.assertEqual(real_conf.communication.fp16, True)
    self.assertEqual(real_conf.communication.fp16_scale, 256)
    self.assertEqual(real_conf.communication.num_communicators, 19)
    self.assertEqual(real_conf.io.drop_last_files, True)
    self.assertEqual(real_conf.io.unbalanced_io_slicing, False)
    self.assertEqual(real_conf.io.slicing, True)
    self.assertEqual(real_conf.cluster.run_visible_devices, "0,4,7")
    self.assertEqual(real_conf.communication.gradients_reduce_method, "sum")

    del os.environ["EPL_COMMUNICATION_SPARSE_AS_DENSE"]
    del os.environ["EPL_COMMUNICATION_MAX_SPLITS"]
    del os.environ["EPL_COMMUNICATION_FP16"]
    del os.environ["EPL_COMMUNICATION_FP16_SCALE"]
    del os.environ["EPL_COMMUNICATION_NUM_COMMUNICATORS"]
    del os.environ["EPL_IO_DROP_LAST_FILES"]
    del os.environ["EPL_IO_SLICING"]
    del os.environ["EPL_CLUSTER_RUN_VISIBLE_DEVICES"]
    del os.environ["EPL_COMMUNICATION_GRADIENTS_REDUCE_METHOD"]

  def test_epl_init_with_env_and_specified_value(self):
    """Test config value init from Env and config."""
    os.environ["EPL_COMMUNICATION_SPARSE_AS_DENSE"] = "True"
    os.environ["EPL_COMMUNICATION_MAX_SPLITS"] = "24"
    os.environ["EPL_COMMUNICATION_FP16"] = "True"
    os.environ["EPL_COMMUNICATION_FP16_SCALE"] = "256"
    os.environ["EPL_IO_DROP_LAST_FILES"] = "True"
    os.environ["EPL_CLUSTER_RUN_VISIBLE_DEVICES"] = "0,4,7"
    os.environ["EPL_COMMUNICATION_GRADIENTS_REDUCE_METHOD"] = "mean"

    conf = Config({
        "communication.max_splits": 15,
        "communication.fp16_scale": 32,
        "communication.sparse_as_dense": True,
        "io.unbalanced_io_slicing": True,
        "communication.gradients_reduce_method": "sum"
    })
    init(conf)
    real_conf = Env.get().config
    self.assertEqual(real_conf.communication.sparse_as_dense, True)
    self.assertEqual(real_conf.communication.max_splits, 15)
    self.assertEqual(real_conf.communication.fp16, True)
    self.assertEqual(real_conf.communication.fp16_scale, 32)
    self.assertEqual(real_conf.communication.num_communicators, 2)
    self.assertEqual(real_conf.io.drop_last_files, True)
    self.assertEqual(real_conf.io.unbalanced_io_slicing, True)
    self.assertEqual(real_conf.cluster.run_visible_devices, "0,4,7")
    self.assertEqual(real_conf.communication.gradients_reduce_method, "sum")

  def test_gradient_checkpoint_from_env(self):
    os.environ["EPL_GRADIENT_CHECKPOINT_TYPE"] = constant.GC_COLLECTION
    conf = Config()
    self.assertEqual(conf.gradient_checkpoint.type, constant.GC_COLLECTION)

    os.environ["EPL_GRADIENT_CHECKPOINT_TYPE"] = constant.GC_AUTO
    conf = Config()
    self.assertEqual(conf.gradient_checkpoint.type, constant.GC_AUTO)

    os.environ["EPL_GRADIENT_CHECKPOINT_END_TASKGRAPH"] = "5"
    conf = Config()
    self.assertEqual(conf.gradient_checkpoint.end_taskgraph, 5)

  def test_reduce_method_exception(self):
    os.environ["EPL_COMMUNICATION_GRADIENTS_REDUCE_METHOD"] = "not_mean"
    with self.assertRaises(ValueError):
      Config()
      self.fail('Test failed, value error expected.')
    del os.environ["EPL_COMMUNICATION_GRADIENTS_REDUCE_METHOD"]

    with self.assertRaises(ValueError):
      Config({"communication.gradients_reduce_method": "not_sum"})
      self.fail('Test failed, value error expected.')


# pylint: enable=missing-docstring,unused-argument

if __name__ == "__main__":
  test.main()
