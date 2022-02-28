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
"""Test for cluster."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python.platform import test
import epl
from epl.cluster import Cluster
# from epl.utils.constant import ENV_VISIBLE_DEVICES


# pylint: disable=missing-docstring,unused-argument,unused-variable
# pylint: disable=protected-access
class ClusterTest(test.TestCase):
  def test_cluster_with_visible_devices(self):
    os.environ["TF_CONFIG"] = \
      '''{"cluster":{"worker":["127.0.0.1:9001","127.0.0.1:9002"]},
          "task":{"type":"worker","index":1}}'''
    os.environ["EPL_CLUSTER_RUN_VISIBLE_DEVICES"] = "1"
    epl.init()
    clus = Cluster()
    self.assertEqual(clus.worker_index, 1)
    self.assertEqual(clus.worker_num, 2)
    self.assertEqual(clus.gpu_num_per_worker, 1)
    self.assertEqual(clus.total_gpu_num, 2)
    self.assertListEqual(clus._cluster_spec.job_tasks("worker"),
                         ["127.0.0.1:9001", "127.0.0.1:9002"])
    self.assertListEqual(clus.available_devices,
                         ["/job:worker/replica:0/task:1/device:GPU:0"])
    self.assertListEqual(clus.virtual_devices[0]._slice_devices,
                         [["/job:worker/replica:0/task:0/device:GPU:0"],
                          ["/job:worker/replica:0/task:1/device:GPU:0"]])


# pylint: enable=missing-docstring,unused-argument,unused-variable
# pylint: enable=protected-access

if __name__ == '__main__':
  test.main()
