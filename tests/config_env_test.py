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


import inspect
from tensorflow.python.platform import test
from epl import init
from epl.cluster import Cluster
from epl.config import Config
from epl.env import Env
from epl.utils.common import get_device_string

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


class RunConfigEnvTest(test.TestCase):
  """Test for epl config."""

  def test_epl_init_with_env_and_specified_value(self):
    """Test config value init from Env and config."""
    init()

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

    config = Config({"communication.num_communicators": 3,
                     "communication.sparse_as_dense": False})
    init(config)
    real_conf = Env.get().config
    self.assertEqual(real_conf.communication.sparse_as_dense, False)
    self.assertEqual(real_conf.communication.max_splits, 24)
    self.assertEqual(real_conf.communication.fp16, True)
    self.assertEqual(real_conf.communication.fp16_scale, 256)
    self.assertEqual(real_conf.communication.num_communicators, 3)
    self.assertEqual(real_conf.io.drop_last_files, True)
    self.assertEqual(real_conf.io.unbalanced_io_slicing, False)
    self.assertEqual(real_conf.io.slicing, True)
    self.assertEqual(real_conf.cluster.run_visible_devices, "0,4,7")
    self.assertEqual(real_conf.communication.gradients_reduce_method, "sum")



  def test_config_docs(self):
    # TODO(sayang): check en api config file.
    doc_file = "../docs/zh/api/config.md"
    with open(doc_file, "rb") as f:
      doc_text = f.read().decode("UTF-8")
    config = Config()
    def _get_attributes(cls):
      return [(name, attr) for name, attr in inspect.getmembers(cls) if not name.startswith('_')]

    for name, conf in _get_attributes(config):
      for sub_name, _ in _get_attributes(conf):
        # e.g. pipeline.num_micro_batch
        config_name = name + '.' + sub_name
        self.assertTrue(config_name in doc_text, "config {} not in doc.".format(config_name))


# pylint: enable=missing-docstring,unused-argument

if __name__ == "__main__":
  test.main()
