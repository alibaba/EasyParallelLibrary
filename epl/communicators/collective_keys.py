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
# ==============================================================================
"""Utilities for collective_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.framework import device as pydev

_lock = threading.Lock()
_thread_local = threading.local()

# Code from tensorflow-1.15, distribute/cross_device_utils.py,
# using for compatible with tensorflow-1.12.
class CollectiveKeys(object):
  """Class that manages collective keys.

  We need to manage three different keys for collective:

  *Group key*: an integer key to identify the set of cooperative devices.
  Collective ops work under the same set of devices must using the same group
  key.

  *Instance key*: an integer key to identify the set of same counterpart of
  tensors on different devices in a device group that need to be all-reduced.

  "Graph key": an integer key that is unique key graph. This is used to support
  multiple graphs per client session. It must be non-zero and set in the
  `config` argument of each call to `session.run`.
  """

  def __init__(self,
               group_key_start=1,
               op_instance_key_start=100,
               variable_instance_key_start=1000000):
    """Initializes the object.

    Args:
      group_key_start: the starting integer of group key.
      op_instance_key_start: the starting integer of instance key for ops.
      variable_instance_key_start: the starting integer of instance key for
        variables.
    """
    self._group_key = group_key_start
    self._group_key_table = {}

    assert op_instance_key_start != variable_instance_key_start
    self._op_instance_key_start = op_instance_key_start
    self._variable_instance_key = variable_instance_key_start

  def _get_thread_local_object(self):
    if not hasattr(_thread_local, 'op_instance_key'):
      _thread_local.op_instance_key = self._op_instance_key_start
    return _thread_local

  def get_group_key(self, devices):
    """Returns a group key for the set of devices.

    Args:
      devices: list of strings naming devices in a collective group.

    Returns:
      int key uniquely identifying the set of device names.
    """
    parsed = [pydev.DeviceSpec.from_string(d) for d in devices]
    # In the between-graph replicated training, different workers need to get
    # the same device key. So we remove the task_type and task_id from the
    # devices.
    names = sorted(['%s:%d' % (d.device_type, d.device_index) for d in parsed])
    key_id = ','.join(names)
    with _lock:
      if key_id not in self._group_key_table:
        new_key = self._group_key
        self._group_key += 1
        self._group_key_table[key_id] = new_key
    return self._group_key_table[key_id]

  def get_op_instance_key(self):
    """Returns a new instance key for use in defining a collective op."""
    v = self._get_thread_local_object().op_instance_key
    self._get_thread_local_object().op_instance_key += 1
    return v

  def get_variable_instance_key(self):
    """Returns a new instance key for use in creating a Variable."""
    v = self._variable_instance_key
    self._variable_instance_key += 1
    return v
