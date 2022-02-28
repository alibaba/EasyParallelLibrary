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

from tensorflow.python.platform import test
from epl.cluster import Cluster
from epl.utils.common import get_device_string

_GPU_PER_WORKER = 1


# pylint: disable=missing-docstring,unused-argument,unused-variable,
# pylint: disable=protected-access
def _mock_available_gpus():
  def available_gpus(self, *args, **kwargs):
    devices = []
    for gpu_index in range(_GPU_PER_WORKER):
      devices.append(get_device_string(task=1, device_index=gpu_index))
    return devices

  return available_gpus


Cluster.available_gpus = _mock_available_gpus()

def get_slices(cluster):
  slices = []
  for vd in cluster.virtual_devices:
    slices.append(vd._slice_devices)
  return slices


class ClusterTest(test.TestCase):
  def test_cluster_with_aware_row(self):
    # 2 workers not in the same machine
    clus = Cluster(worker_hosts="192.34.5.1:8000,192.34.5.2:8001",
                   worker_index=1,
                   layout={"aware_row": 1})
    self.assertEqual(clus.worker_index, 1)
    self.assertEqual(clus.worker_num, 2)
    self.assertEqual(clus.total_gpu_num, clus.worker_num)
    slices = get_slices(clus)
    self.assertEqual(len(slices), 2)

    self.assertEqual(len(slices[0]), 1)
    self.assertEqual(slices[0][0][0],
                     "/job:worker/replica:0/task:0/device:GPU:0")

    self.assertEqual(len(slices[1]), 1)
    self.assertEqual(slices[1][0][0],
                     "/job:worker/replica:0/task:1/device:GPU:0")

    # 4 workers
    hosts = "192.34.5.1:8000,192.34.5.2:8001,192.34.5.1:8002,192.34.5.2:8003"
    clus = Cluster(worker_hosts=hosts, worker_index=1, layout={"aware_row": 2})
    self.assertEqual(clus.worker_index, 2)
    self.assertEqual(clus.worker_num, 4)
    self.assertEqual(clus.total_gpu_num, clus.worker_num)
    slices = get_slices(clus)
    self.assertEqual(len(slices), 2)

    self.assertEqual(len(slices[0]), 2)
    self.assertEqual(slices[0][0][0],
                     "/job:worker/replica:0/task:0/device:GPU:0")
    self.assertEqual(slices[0][1][0],
                     "/job:worker/replica:0/task:1/device:GPU:0")

    self.assertEqual(len(slices[1]), 2)
    self.assertEqual(slices[1][0][0],
                     "/job:worker/replica:0/task:2/device:GPU:0")
    self.assertEqual(slices[1][1][0],
                     "/job:worker/replica:0/task:3/device:GPU:0")

    # Test for un-divisible scene.
    with self.assertRaises(RuntimeError):
      clus = Cluster(worker_hosts="127.0.0.1:8000,127.0.0.1:8001",
                     worker_index=1,
                     layout={"aware_row": 3})


# pylint: enable=missing-docstring,unused-argument,unused-variable,
# pylint: enable=protected-access

if __name__ == '__main__':
  test.main()
