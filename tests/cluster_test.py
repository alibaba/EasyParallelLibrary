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
from epl.cluster import Cluster
from epl.utils.common import get_device_string

_GPU_PER_WORKER = 4


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
  def test_cluster_without_tf_config(self):
    os.environ["TF_CONFIG"] = ""
    clus = Cluster()
    self.assertEqual(clus.worker_index, 0)
    self.assertEqual(clus.worker_num, 1)
    self.assertEqual(clus.gpu_num_per_worker, _GPU_PER_WORKER)
    self.assertEqual(clus.total_gpu_num, _GPU_PER_WORKER)

  def test_cluster_with_tf_config(self):
    os.environ["TF_CONFIG"] = \
      '''{"cluster":{"worker":["127.0.0.1:9001","127.0.0.1:9002"]},
          "task":{"type":"worker","index":1}}'''
    clus = Cluster()
    self.assertEqual(clus.worker_index, 1)
    self.assertEqual(clus.worker_num, 2)
    self.assertEqual(clus.gpu_num_per_worker, _GPU_PER_WORKER)
    self.assertEqual(clus.total_gpu_num, _GPU_PER_WORKER * 2)
    self.assertListEqual(clus._cluster_spec.job_tasks("worker"),
                         ["127.0.0.1:9001", "127.0.0.1:9002"])

    os.environ["TF_CONFIG"] = \
      '''{"cluster":{"worker":["127.0.0.1:9001","127.0.0.1:9002"],
          "ps":["127.0.0.1:9003","127.0.0.1:9004"]},
          "task":{"type":"ps","index":1}}'''
    clus = Cluster()
    self.assertEqual(clus.worker_index, 3)
    self.assertEqual(clus.worker_num, 4)
    self.assertEqual(clus.gpu_num_per_worker, _GPU_PER_WORKER)
    self.assertEqual(clus.total_gpu_num, _GPU_PER_WORKER * 4)
    self.assertListEqual(clus._cluster_spec.job_tasks("worker"), [
        "127.0.0.1:9001", "127.0.0.1:9002", "127.0.0.1:9003", "127.0.0.1:9004"
    ])

    os.environ["TF_CONFIG"] = \
      '''{"cluster":{"worker":["127.0.0.1:9001","127.0.0.1:9002"],
          "ps":["127.0.0.1:9003","127.0.0.1:9004"],
          "chief":["127.0.0.1:9005"]},
          "task":{"type":"worker","index":1}}'''
    clus = Cluster()
    self.assertEqual(clus.worker_index, 2)
    self.assertEqual(clus.worker_num, 5)
    self.assertEqual(clus.gpu_num_per_worker, _GPU_PER_WORKER)
    self.assertEqual(clus.total_gpu_num, _GPU_PER_WORKER * 5)
    self.assertListEqual(clus._cluster_spec.job_tasks("worker"), [
        "127.0.0.1:9005", "127.0.0.1:9001", "127.0.0.1:9002", "127.0.0.1:9003",
        "127.0.0.1:9004"
    ])

    os.environ["TF_CONFIG"] = \
      '''{"cluster":{"worker":["127.0.0.1:9001","127.0.0.1:9002"],
          "ps":["127.0.0.1:9003","127.0.0.1:9004"],
          "chief":["127.0.0.1:9005"]},
          "task":{"type":"chief","index":0}}'''
    clus = Cluster()
    self.assertEqual(clus.worker_index, 0)
    self.assertEqual(clus.worker_num, 5)
    self.assertEqual(clus.gpu_num_per_worker, _GPU_PER_WORKER)
    self.assertEqual(clus.total_gpu_num, _GPU_PER_WORKER * 5)
    self.assertListEqual(clus._cluster_spec.job_tasks("worker"), [
        "127.0.0.1:9005", "127.0.0.1:9001", "127.0.0.1:9002", "127.0.0.1:9003",
        "127.0.0.1:9004"
    ])

    os.environ["TF_CONFIG"] = \
      '''{"cluster":{"chief":["127.0.0.1:9001"]},
          "task":{"type":"chief","index":0}}'''
    clus = Cluster()
    self.assertEqual(clus.worker_index, 0)
    self.assertEqual(clus.worker_num, 1)
    self.assertEqual(clus.gpu_num_per_worker, _GPU_PER_WORKER)
    self.assertEqual(clus.total_gpu_num, _GPU_PER_WORKER)
    self.assertListEqual(clus._cluster_spec.job_tasks("worker"),
                         ["127.0.0.1:9001"])

    del os.environ["TF_CONFIG"]

  def test_unkown_slice_layout(self):
    with self.assertRaises(RuntimeError):
      clus = Cluster(worker_hosts="127.0.0.1:8000,127.0.0.1:8001",
                     worker_index=1,
                     layout={"unkown_layout": 2})

  def test_cluster_with_all(self):
    clus = Cluster(worker_hosts="127.0.0.1:8000,127.0.0.1:8001",
                   worker_index=0,
                   layout="all")
    self.assertEqual(clus.worker_index, 0)
    self.assertEqual(clus.worker_num, 2)
    self.assertEqual(clus.gpu_num_per_worker, _GPU_PER_WORKER)
    self.assertEqual(clus.total_gpu_num, _GPU_PER_WORKER * 2)
    slices = get_slices(clus)

    self.assertEqual(len(slices), 1)

    self.assertEqual(len(slices[0]), _GPU_PER_WORKER * 2)
    self.assertEqual(slices[0][0][0],
                     "/job:worker/replica:0/task:0/device:GPU:0")
    self.assertEqual(slices[0][1][0],
                     "/job:worker/replica:0/task:0/device:GPU:1")
    self.assertEqual(slices[0][2][0],
                     "/job:worker/replica:0/task:0/device:GPU:2")
    self.assertEqual(slices[0][3][0],
                     "/job:worker/replica:0/task:0/device:GPU:3")
    self.assertEqual(slices[0][4][0],
                     "/job:worker/replica:0/task:1/device:GPU:0")
    self.assertEqual(slices[0][5][0],
                     "/job:worker/replica:0/task:1/device:GPU:1")
    self.assertEqual(slices[0][6][0],
                     "/job:worker/replica:0/task:1/device:GPU:2")
    self.assertEqual(slices[0][7][0],
                     "/job:worker/replica:0/task:1/device:GPU:3")


  def test_cluster_with_specific(self):
    clus = Cluster(worker_hosts="127.0.0.1:8000,127.0.0.1:8001",
                   worker_index=1,
                   layout={
                       "specific":
                       [[["/job:worker/replica:0/task:0/device:GPU:0"],
                         ["/job:worker/replica:0/task:1/device:GPU:0"],
                         ["/job:worker/replica:0/task:0/device:GPU:1"],
                         ["/job:worker/replica:0/task:1/device:GPU:1"]],
                        [["/job:worker/replica:0/task:0/device:GPU:2"],
                         ["/job:worker/replica:0/task:1/device:GPU:2"],
                         ["/job:worker/replica:0/task:0/device:GPU:3"],
                         ["/job:worker/replica:0/task:1/device:GPU:3"]]]
                   })
    self.assertEqual(clus.worker_index, 1)
    self.assertEqual(clus.worker_num, 2)
    self.assertEqual(clus.gpu_num_per_worker, _GPU_PER_WORKER)
    self.assertEqual(clus.total_gpu_num, _GPU_PER_WORKER * 2)
    slices = get_slices(clus)
    self.assertEqual(len(slices), 2)

    self.assertEqual(len(slices[0]), 4)
    self.assertListEqual(slices[0],
                         [["/job:worker/replica:0/task:0/device:GPU:0"],
                          ["/job:worker/replica:0/task:1/device:GPU:0"],
                          ["/job:worker/replica:0/task:0/device:GPU:1"],
                          ["/job:worker/replica:0/task:1/device:GPU:1"]])
    self.assertEqual(len(slices[1]), 4)
    self.assertListEqual(slices[1],
                         [["/job:worker/replica:0/task:0/device:GPU:2"],
                          ["/job:worker/replica:0/task:1/device:GPU:2"],
                          ["/job:worker/replica:0/task:0/device:GPU:3"],
                          ["/job:worker/replica:0/task:1/device:GPU:3"]])
    self.assertEqual(clus.virtual_devices[0].local_devices,
                     ('/job:worker/replica:0/task:1/device:GPU:0', '/job:worker/replica:0/task:1/device:GPU:1'))
    self.assertEqual(clus.virtual_devices[1].local_devices,
                     ('/job:worker/replica:0/task:1/device:GPU:2', '/job:worker/replica:0/task:1/device:GPU:3'))

  def test_cluster_with_ps(self):
    clus = Cluster(worker_hosts="127.0.0.1:8000,127.0.0.1:8001",
                   ps_hosts="127.0.0.1:8002,127.0.0.1:8003",
                   job_name="ps",
                   worker_index=0,
                   layout="all")
    self.assertEqual(clus.worker_index, 2)
    self.assertEqual(clus.worker_num, 4)
    self.assertEqual(clus.gpu_num_per_worker, _GPU_PER_WORKER)
    self.assertEqual(clus.total_gpu_num, _GPU_PER_WORKER * 4)
    slices = get_slices(clus)
    self.assertEqual(len(slices), 1)

    self.assertEqual(len(slices[0]), _GPU_PER_WORKER * 4)
    self.assertEqual(slices[0][0][0],
                     "/job:worker/replica:0/task:0/device:GPU:0")
    self.assertEqual(slices[0][1][0],
                     "/job:worker/replica:0/task:0/device:GPU:1")
    self.assertEqual(slices[0][2][0],
                     "/job:worker/replica:0/task:0/device:GPU:2")
    self.assertEqual(slices[0][3][0],
                     "/job:worker/replica:0/task:0/device:GPU:3")
    self.assertEqual(slices[0][4][0],
                     "/job:worker/replica:0/task:1/device:GPU:0")
    self.assertEqual(slices[0][5][0],
                     "/job:worker/replica:0/task:1/device:GPU:1")
    self.assertEqual(slices[0][6][0],
                     "/job:worker/replica:0/task:1/device:GPU:2")
    self.assertEqual(slices[0][7][0],
                     "/job:worker/replica:0/task:1/device:GPU:3")
    self.assertEqual(slices[0][8][0],
                     "/job:worker/replica:0/task:2/device:GPU:0")
    self.assertEqual(slices[0][9][0],
                     "/job:worker/replica:0/task:2/device:GPU:1")
    self.assertEqual(slices[0][10][0],
                     "/job:worker/replica:0/task:2/device:GPU:2")
    self.assertEqual(slices[0][11][0],
                     "/job:worker/replica:0/task:2/device:GPU:3")
    self.assertEqual(slices[0][12][0],
                     "/job:worker/replica:0/task:3/device:GPU:0")
    self.assertEqual(slices[0][13][0],
                     "/job:worker/replica:0/task:3/device:GPU:1")
    self.assertEqual(slices[0][14][0],
                     "/job:worker/replica:0/task:3/device:GPU:2")
    self.assertEqual(slices[0][15][0],
                     "/job:worker/replica:0/task:3/device:GPU:3")


# pylint: enable=missing-docstring,unused-argument,unused-variable,
# pylint: enable=protected-access

if __name__ == '__main__':
  test.main()
