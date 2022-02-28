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
"""Test for utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import random

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test

from epl.utils import common
from epl.parallel import partitioner
from epl.communicators.collective_communicator import estimate_split_num_for_comm
from epl.parallel.graph_editor import get_global_gcd_from_dict
from epl.parallel.graph_editor import fetch_slice_objects_proportion_to_local_num_replicas

# pylint: disable=missing-docstring,unused-argument,unused-variable
# pylint: disable=invalid-name
class UtilsTest(test.TestCase):
  def test_device_string(self):
    self.assertEqual(common.get_device_string(),
                     "/job:worker/replica:0/task:0/device:GPU:0")
    self.assertEqual(common.get_device_string(device_type="CPU"),
                     "/job:worker/replica:0/task:0/device:CPU:0")
    self.assertEqual(common.get_device_string(job="ps"),
                     "/job:ps/replica:0/task:0/device:GPU:0")
    self.assertEqual(common.get_device_string(task=1),
                     "/job:worker/replica:0/task:1/device:GPU:0")
    self.assertEqual(common.get_device_string(replica=2),
                     "/job:worker/replica:2/task:0/device:GPU:0")
    self.assertEqual(common.get_device_string(device_index=2),
                     "/job:worker/replica:0/task:0/device:GPU:2")
    self.assertEqual(common.get_device_string(task=1, device_index=2),
                     "/job:worker/replica:0/task:1/device:GPU:2")

  def test_get_task_index_from_device_str(self):
    device_str = "/job:worker/replica:0/task:0/device:GPU:0"
    self.assertEqual(0, common.get_task_index_from_device_str(device_str))
    device_str = "/job:worker/replica:0/task:1/device:GPU:2"
    self.assertEqual(1, common.get_task_index_from_device_str(device_str))
    device_str = "/job:worker/replica:0/task:2/device:CPU:0"
    self.assertEqual(2, common.get_task_index_from_device_str(device_str))

  def test_get_replica_prefix(self):
    self.assertEqual("", common.get_replica_prefix(0))
    self.assertEqual("EPL_REPLICA_1/", common.get_replica_prefix(1))

  def test_get_micro_batch_prefix(self):
    self.assertEqual("", common.get_micro_batch_prefix(0))
    self.assertEqual("EPL_MICRO_BATCH_1/", common.get_micro_batch_prefix(1))

  def test_get_replica_prefix_from_node_name(self):
    op_name = "dense/MatMul"
    self.assertEqual("", common.get_replica_prefix_from_node_name(op_name))
    op_name = "EPL_REPLICA_1/dense/MatMul"
    self.assertEqual("EPL_REPLICA_1/",
                     common.get_replica_prefix_from_node_name(op_name))
    op_name = "EPL_MICRO_BATCH_1/dense/MatMul"
    self.assertEqual("", common.get_replica_prefix_from_node_name(op_name))
    op_name = "EPL_REPLICA_1/EPL_MICRO_BATCH_1/dense/MatMul"
    self.assertEqual("EPL_REPLICA_1/",
                     common.get_replica_prefix_from_node_name(op_name))

  def test_get_micro_batch_prefix_from_node_name(self):
    op_name = "dense/MatMul"
    self.assertEqual("", common.get_micro_batch_prefix_from_node_name(op_name))
    op_name = "EPL_REPLICA_1/dense/MatMul"
    self.assertEqual("", common.get_micro_batch_prefix_from_node_name(op_name))
    op_name = "EPL_MICRO_BATCH_1/dense/MatMul"
    self.assertEqual("EPL_MICRO_BATCH_1/",
                     common.get_micro_batch_prefix_from_node_name(op_name))
    op_name = "EPL_REPLICA_1/EPL_MICRO_BATCH_1/dense/MatMul"
    self.assertEqual("EPL_MICRO_BATCH_1/",
                     common.get_micro_batch_prefix_from_node_name(op_name))

  def test_get_replica_index_from_node_name(self):
    op_name = "dense/MatMul"
    self.assertEqual(0, common.get_replica_index_from_node_name(op_name))
    op_name = "EPL_MICRO_BATCH_1/dense/MatMul"
    self.assertEqual(0, common.get_replica_index_from_node_name(op_name))
    op_name = "EPL_REPLICA_1/dense/MatMul"
    self.assertEqual(1, common.get_replica_index_from_node_name(op_name))
    op_name = "EPL_REPLICA_1/EPL_MICRO_BATCH_1/dense/MatMul"
    self.assertEqual(1, common.get_replica_index_from_node_name(op_name))

  def test_get_micro_batch_index_from_node_name(self):
    op_name = "dense/MatMul"
    self.assertEqual(0, common.get_micro_batch_index_from_node_name(op_name))
    op_name = "EPL_REPLICA_1/dense/MatMul"
    self.assertEqual(0, common.get_micro_batch_index_from_node_name(op_name))
    op_name = "EPL_MICRO_BATCH_1/dense/MatMul"
    self.assertEqual(1, common.get_micro_batch_index_from_node_name(op_name))
    op_name = "EPL_REPLICA_1/EPL_MICRO_BATCH_1/dense/MatMul"
    self.assertEqual(1, common.get_micro_batch_index_from_node_name(op_name))

  def test_get_original_name_from_cloned_object(self):
    orig_name = "dense/MatMul"
    op_name = "dense/MatMul"
    self.assertEqual(orig_name,
                     common.get_original_name_from_cloned_object(op_name))
    op_name = "EPL_REPLICA_1/dense/MatMul"
    self.assertEqual(orig_name,
                     common.get_original_name_from_cloned_object(op_name))
    op_name = "EPL_MICRO_BATCH_1/dense/MatMul"
    self.assertEqual(orig_name,
                     common.get_original_name_from_cloned_object(op_name))
    op_name = "EPL_REPLICA_1/EPL_MICRO_BATCH_1/dense/MatMul"
    self.assertEqual(orig_name,
                     common.get_original_name_from_cloned_object(op_name))
    op_name = "EPL_REPLICA_11/EPL_MICRO_BATCH_123/dense/MatMul"
    self.assertEqual(orig_name,
                     common.get_original_name_from_cloned_object(op_name))

  def test_for_update_tuple(self):
    # Test for update tuple.
    t1 = ("str1", 123, 456)
    update_value = "str2"
    update_index = 0
    t1 = common.update_tuple(t1, update_value, update_index)
    self.assertEqual(t1[update_index], update_value)
    self.assertEqual(t1[1], 123)
    self.assertEqual(t1[2], 456)

    update_value = 789
    update_index = 2
    t2 = ("str1", 123, 456)
    t2 = common.update_tuple(t2, update_value, update_index)
    self.assertEqual(t2[0], "str1")
    self.assertEqual(t2[1], 123)
    self.assertEqual(t2[update_index], 789)

    # Test for out of range error.
    update_value = "str2"
    update_index = 3
    t3 = ("str1", 123, 456)
    with self.assertRaises(ValueError):
      t3 = common.update_tuple(t3, update_value, update_index)

    # Test for namedtuple.
    NTuple = namedtuple("NTuple", ["field0", "field1", "field2"])

    ntuple1 = NTuple(field0="str1", field1=123, field2=456)
    update_index = 0
    update_value = "str2"
    ntuple1 = common.update_tuple(ntuple1, update_value, update_index)
    self.assertEqual(ntuple1.field0, update_value)
    self.assertEqual(ntuple1.field1, 123)
    self.assertEqual(ntuple1.field2, 456)

    update_index = 2
    update_value = 789
    ntuple2 = NTuple(field0="str1", field1=123, field2=456)
    ntuple2 = common.update_tuple(ntuple2, update_value, update_index)
    self.assertEqual(ntuple2.field0, "str1")
    self.assertEqual(ntuple2.field1, 123)
    self.assertEqual(ntuple2.field2, update_value)

    # Test for out of range error.
    update_index = 3
    update_value = 789
    ntuple3 = NTuple(field0="str1", field1=123, field2=456)
    with self.assertRaises(ValueError):
      ntuple3 = common.update_tuple(ntuple3, update_value, update_index)

  def test_for_global_gcd(self):
    """Test for fetch global gcd for all values in a dict."""
    original_dict = dict()
    original_dict[0] = 8
    original_dict[1] = 4
    original_dict[2] = 2
    global_gcd = get_global_gcd_from_dict(original_dict)
    self.assertEqual(global_gcd, 2)

  def test_fetch_slice_objects_proportion_to_local_num_replicas(self):
    """Test objects slicing proportion to local num replicas."""
    # Scene 1 : test balanced num_local_replicas across constructors
    all_devices = \
        ["/job:worker/replica:0/task:0/device:GPU:0",
         "/job:worker/replica:0/task:0/device:GPU:1",
         "/job:worker/replica:0/task:1/device:GPU:0",
         "/job:worker/replica:0/task:1/device:GPU:1"]
    num_replicas = 4
    # Test files can be proportionately averaged
    obj_list = ["file_0", "file_1"]
    slice_rank_0 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            0, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_0, ["file_0"])
    slice_rank_1 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            1, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_1, ["file_1"])

    obj_list = ["file_0", "file_1", "file_2", "file_3"]
    slice_rank_0 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            0, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_0, ["file_0", "file_2"])
    slice_rank_1 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            1, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_1, ["file_1", "file_3"])

    # Test files can not be proportionately averaged
    # and maybe some constructor handles no file
    obj_list = ["file_0"]
    slice_rank_0 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            0, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_0, ["file_0"])
    slice_rank_1 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            1, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_1, ["file_0"])

    # Test files can not be proportionately averaged
    # and file is enough
    obj_list = ["file_0", "file_1", "file_2"]
    slice_rank_0 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            0, obj_list, num_replicas,
            all_devices, True)
    self.assertEqual(slice_rank_0, ["file_0"])
    slice_rank_0 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            0, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_0, ["file_0", "file_1", "file_2"])
    slice_rank_1 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            1, obj_list, num_replicas,
            all_devices, True)
    self.assertEqual(slice_rank_1, ["file_1"])
    slice_rank_1 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            1, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_1, ["file_0", "file_1", "file_2"])

    # Test unbalanced num_local_replicas between constructors
    all_devices = \
        ["/job:worker/replica:0/task:0/device:GPU:0",
         "/job:worker/replica:0/task:0/device:GPU:1",
         "/job:worker/replica:0/task:0/device:GPU:2",
         "/job:worker/replica:0/task:0/device:GPU:3",
         "/job:worker/replica:0/task:1/device:GPU:0",
         "/job:worker/replica:0/task:1/device:GPU:1"]
    num_replicas = 6

    # Test files can be proportionately averaged
    obj_list = ["file_0", "file_1", "file_2"]
    slice_rank_0 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            0, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_0, ["file_0", "file_1"])
    slice_rank_1 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            1, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_1, ["file_2"])

    # Test files can not be proportionately averaged
    # and maybe some constructor handles no file
    obj_list = ["file_0", "file_1"]
    slice_rank_0 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            0, obj_list, num_replicas,
            all_devices)
    list.sort(slice_rank_0)
    self.assertEqual(slice_rank_0, ["file_0", "file_0", "file_1", "file_1"])
    slice_rank_1 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            1, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_1, ["file_0", "file_1"])

    # Test files can not be proportionately averaged
    # and file is enough
    obj_list = ["file_0", "file_1", "file_2", "file_3"]
    slice_rank_0 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            0, obj_list, num_replicas,
            all_devices, True)
    self.assertEqual(slice_rank_0, ["file_0", "file_1"])
    slice_rank_0 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            0, obj_list, num_replicas,
            all_devices)
    list.sort(slice_rank_0)
    self.assertEqual(slice_rank_0,
                     ["file_0", "file_0", "file_1", "file_1", \
                      "file_2", "file_2", "file_3", "file_3"])
    slice_rank_0 = \
      fetch_slice_objects_proportion_to_local_num_replicas(
          0, obj_list, num_replicas,
          all_devices, False, True)
    list.sort(slice_rank_0)
    self.assertEqual(slice_rank_0, ["file_0", "file_1", "file_3"])

    slice_rank_1 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            1, obj_list, num_replicas,
            all_devices, True)
    self.assertEqual(slice_rank_1, ["file_2"])
    slice_rank_1 = \
        fetch_slice_objects_proportion_to_local_num_replicas(
            1, obj_list, num_replicas,
            all_devices)
    self.assertEqual(slice_rank_1, ["file_0", "file_1", "file_2", "file_3"])
    slice_rank_1 = \
      fetch_slice_objects_proportion_to_local_num_replicas(
          1, obj_list, num_replicas,
          all_devices, False, True)
    self.assertEqual(slice_rank_1, ["file_2"])

  def test_partition_stages(self):
    res = partitioner.partition_stages(['a']*10, [1]*10, 2)
    self.assertEqual(res, [['a']*5, ['a']*5])
    res = partitioner.partition_stages(['a']*10, [1]*10, 5)
    self.assertEqual(res, [['a']*2 for i in range(5)])
    for i in range(1, 1024, 8):
      data = ['a'] * 10
      weights = [1] * len(data)
      res = partitioner.partition_stages(data, weights, i)
      self.assertEqual(len(res), i)
      flatten = [i for sub in res for i in sub]
      self.assertEqual(data, flatten)
    for i in range(1, 1024, 8):
      data = [random.randrange(1, 100, 1) for i in range(512)]
      res = partitioner.partition_stages(data, data, i)
      self.assertEqual(len(res), i)
      flatten = [i for sub in res for i in sub]
      self.assertEqual(data, flatten)

  def test_partition_stages_uneven(self):
    data = ['a', 'b', 'c', 'd', 'e', 'f']
    weights = [1024, 1, 2, 3, 4, 5]
    res = partitioner.partition_stages(data, weights, 4)
    self.assertEqual(len(res), 4)
    flatten = [i for sub in res for i in sub]
    self.assertEqual(data, flatten)

  def test_estimate_split_num_for_comm(self):
    # 32MB
    a = constant_op.constant(1, shape=[8, 1024, 1024], dtype=dtypes.int32)
    # 28MB
    b = constant_op.constant(1.0, shape=[14, 1024, 1024], dtype=dtypes.float16)
    # 36MB
    c = constant_op.constant(1.0, shape=[9, 1024, 1024], dtype=dtypes.float32)
    self.assertEqual(estimate_split_num_for_comm([]), 1)
    self.assertEqual(estimate_split_num_for_comm([a]), 1)
    self.assertEqual(estimate_split_num_for_comm(a), 1)
    self.assertEqual(estimate_split_num_for_comm([b]), 1)
    self.assertEqual(estimate_split_num_for_comm([c]), 2)
    self.assertEqual(estimate_split_num_for_comm([a, b]), 2)
    self.assertEqual(estimate_split_num_for_comm([a, b, c]), 4)


# pylint: enable=missing-docstring,unused-argument,unused-variable,line-too-long
# pylint: enable=invalid-name

if __name__ == '__main__':
  test.main()
