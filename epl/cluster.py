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
"""Cluster for grouping devices into serveral slices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import numpy as np
from tensorflow.python.client import device_lib
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import server_lib

from epl.env import Env
from epl.ir.graph import Graph
from epl.utils import common
from epl.utils import constant
from epl.utils import metric


class VirtualDevice(object):
  """Devices for one task graph"""

  def __init__(self, index, slice_devices, worker_index, local_device_indices=None):
    """
    Initialize a VirtualDevice object.

    Args:
      index: virtual device index.
      slice_devices: a list of device tuple.
      worker_index: current worker index.
      local_device_indices: indices to get local indices.
    """
    self._index = index
    self._worker_index = worker_index
    self._slice_devices = slice_devices
    self._all_devices = tuple(np.reshape(slice_devices, [-1]))

    if self._index == 0:
      # constructor
      self._local_device_indices = tuple(idx for idx, device in enumerate(self._all_devices) if self._device_in_local_worker(device))
    else:
      if local_device_indices is None:
        raise RuntimeError("local_device_indices is required for non-constructors")
      self._local_device_indices = local_device_indices
    self._local_devices = tuple(self._all_devices[idx] for idx in self._local_device_indices)

  @property
  def local_devices(self):
    return self._local_devices

  @property
  def local_device_indices(self):
    return self._local_device_indices

  @property
  def all_devices(self):
    return self._all_devices

  def get_device(self, replica_idx, device_idx):
    """
    Args:
      replica_idx: replica index.
      device_idx: device index for certain replica.
    """
    return self._slice_devices[replica_idx][device_idx]

  def _device_in_local_worker(self, device):
    return common.get_task_index_from_device_str(device) == self._worker_index

  @property
  def num_replicas(self):
    """Global number of replicas."""
    return len(self._all_devices) if self._all_devices else 1

  def __str__(self):
    return str(self._slice_devices)

  def __repr__(self):
    return self.__str__()

class LayoutImpl(object):
  @staticmethod
  def slice(layout_info, clus):
    raise NotImplementedError


class AllLayout(LayoutImpl):
  """Make all devices into one slice."""
  @staticmethod
  def slice(layout_info, clus):
    slices = [[]]
    for worker_index in range(clus.worker_num):
      for gpu_index in range(clus.gpu_num_per_worker):
        slices[0].append([
            common.get_device_string(task=worker_index, device_index=gpu_index)
        ])
    return slices


def get_device_list(worker_num, gpu_num_per_worker, prefer_row=True):
  """Get a list of devices from each worker."""
  if prefer_row:
    for worker_index in range(worker_num):
      for gpu_index in range(gpu_num_per_worker):
        yield common.get_device_string(task=worker_index, device_index=gpu_index)
  else:
    for gpu_index in range(gpu_num_per_worker):
      for worker_index in range(worker_num):
        yield common.get_device_string(task=worker_index, device_index=gpu_index)


def generate_device_slices(cluster, device_per_replicas, num_replica):
  """Generate device slices."""
  num_taskgraph = len(device_per_replicas)
  slices = [[[] for _ in range(num_replica)] for _ in range(num_taskgraph)]
  prefer_row = Env.get().config.cluster.device_place_prefer_intra_node
  all_devices = get_device_list(cluster.worker_num, cluster.gpu_num_per_worker, prefer_row)
  for replica_id in range(num_replica):
    for ti in range(num_taskgraph):
      for _ in range(device_per_replicas[ti]):
        slices[ti][replica_id].append(next(all_devices))
  return slices


class AutoLayout(LayoutImpl):
  """Group devices into slices automatically based on taskgraphs."""

  @staticmethod
  def slice(layout_info, clus):
    # TODO(sayang): Support fuse/cross nodes.
    taskgraphs = Graph.get().taskgraphs
    total_device_num = clus.worker_num * clus.gpu_num_per_worker
    num_device_per_replica = sum(tg.num_device_per_replica for tg in taskgraphs)
    if total_device_num % num_device_per_replica != 0:
      raise RuntimeError("Total devices {} is not divisible by num_device_per_replica {}".format(total_device_num, num_device_per_replica))
    num_replica = total_device_num // num_device_per_replica
    device_per_replicas = [tg.num_device_per_replica for tg in taskgraphs]
    return generate_device_slices(clus, device_per_replicas, num_replica)


class SpecificLayout(LayoutImpl):
  """Use slices specified by users."""
  @staticmethod
  def slice(layout_info, clus):
    return layout_info


class AwareRowLayout(LayoutImpl):
  """Slice cluster in row way with aware net topology."""
  @staticmethod
  def slice(layout_info, clus):
    gpu_num_per_slice = int(layout_info)
    total_worker_num = len(clus.hosts.split(","))
    slice_num = int(total_worker_num / gpu_num_per_slice)

    if clus.gpu_num_per_worker != 1 or slice_num == 0 \
       or total_worker_num % slice_num != 0:
      raise RuntimeError(
          "GPU per-worker is not 1, or total GPU number "
          "is not divisible by stage number. Total machine"
          ": %s, GPU per-worker: %s, stage: %s." %
          (total_worker_num, clus.gpu_num_per_worker, slice_num))

    slices = [[] for unused_x in range(slice_num)]
    gpu_index = 0
    for worker_index in range(total_worker_num):
      slice_index = worker_index / gpu_num_per_slice
      slices[int(slice_index)].append(
          [common.get_device_string(task=worker_index, device_index=gpu_index)])
    return slices

  @staticmethod
  def reorder_hosts(clus):
    """ Reorder hosts and reset worker_index."""
    # Group workers by machine
    hosts = clus.hosts.split(",")
    workers_group_by_machine = {}
    for idx_worker, host in enumerate(hosts):
      host_name = host.split(":")[0]
      if host_name not in workers_group_by_machine:
        workers_group_by_machine[host_name] = []
      workers_group_by_machine[host_name].append(idx_worker)

    # Check number of workers for each machine
    worker_num_per_machine = 0
    for key in workers_group_by_machine:
      if worker_num_per_machine == 0:
        worker_num_per_machine = len(workers_group_by_machine[key])
      elif worker_num_per_machine != len(workers_group_by_machine[key]):
        raise RuntimeError("Number of workers must be " \
                           "the same for each machine.")
    clus.worker_num_per_machine = worker_num_per_machine

    # Reorder hosts
    new_hosts = ""
    for key in workers_group_by_machine:
      host_temp = ""
      for worker in workers_group_by_machine[key]:
        if not host_temp:
          host_temp = hosts[worker]
        else:
          host_temp = host_temp + "," + hosts[worker]
      # Rank 0 must be the same after reordering
      if not new_hosts:
        new_hosts = host_temp
      elif workers_group_by_machine[key][0] == 0:
        new_hosts = host_temp + "," + new_hosts
      else:
        new_hosts = new_hosts + "," + host_temp
    tf_logging.info("Reorder hosts by AwareRowLayout,"
                    " origin hosts: {}, new hosts: {}" \
                    .format(clus.hosts, new_hosts))
    clus.hosts = new_hosts

    # Reset worker_index
    new_hosts_list = new_hosts.split(",")
    for idx_worker, host in enumerate(new_hosts_list):
      if hosts[clus.worker_index] == host:
        clus.worker_index = idx_worker
        break


class Layout(object):
  """Layout of slicing cluster to slices."""
  def __init__(self, layout):
    self._data = {}
    if isinstance(layout, str):
      self._data[layout] = "none"
    else:
      self._data = dict(layout)

  def __getattr__(self, name):
    if name in self._data:
      return self._data[name]
    return None

  def __str__(self):
    return str(self._data)

  def __repr__(self):
    return self.__str__()

  def slice(self, clus):
    """Slice cluster to slices."""
    total_state = \
        int(bool(self.specific)) + \
        int(bool(self.all)) + \
        int(bool(self.auto))
    if total_state > 1:
      raise ValueError("Can't set multiple layout to slice cluster. Layout: %s" % self)

    if self.all:
      return AllLayout.slice(self.all, clus)

    if self.specific:
      return SpecificLayout.slice(self.specific, clus)

    if self.aware_row:
      return AwareRowLayout.slice(self.aware_row, clus)

    if self.auto:
      return AutoLayout.slice(self.auto, clus)

    raise RuntimeError("Layout is not supported. Layout: %s ." % self)

  def reorder_hosts(self, clus):
    """Reorder hosts and reset worker_index for cluster."""
    if self.aware_row:
      AwareRowLayout.reorder_hosts(clus)


class Cluster(object):  # pylint: disable=invalid-name
  """epl cluster."""
  def __init__(self,
               worker_hosts=None,
               ps_hosts=None,
               job_name=constant.DEFAULT_TASK_NAME,
               worker_index=0,
               layout="all"):
    # Try to get hosts information from TF_CONFIG if worker hosts is None.
    if not worker_hosts:
      tf_config = os.environ.get(constant.ENV_TF_CONFIG)
      if not tf_config:
        # Construct TF_CONFIG to schedule a server even for one worker.
        tf_logging.info("Training as a single worker for no TF_CONFIG found.")
        # Specifying port 0 means that the OS will choose a free port for the
        # server.
        tf_config = '{"cluster":{"worker":["127.0.0.1:0"]},"task":' \
                    '{"type":"worker","index":0}}'
      tf_config_json = json.loads(tf_config)
      tf_config_worker_hosts = tf_config_json.get("cluster", {}).get("worker")
      tf_config_ps_hosts = tf_config_json.get("cluster", {}).get("ps")
      tf_config_chief_hosts = tf_config_json.get("cluster", {}).get("chief")
      tf_config_job_name = tf_config_json.get("task", {}).get("type")
      tf_config_task_index = tf_config_json.get("task", {}).get("index")

      if tf_config_chief_hosts:
        if not tf_config_worker_hosts:
          tf_config_worker_hosts = tf_config_chief_hosts
        else:
          # Put chief host before worker hosts to treat chief as worker 0.
          tf_config_worker_hosts = \
              tf_config_chief_hosts + tf_config_worker_hosts
        if tf_config_job_name == constant.DEFAULT_TASK_NAME:
          tf_config_task_index += 1
        if tf_config_job_name == constant.CHIEF_WORKER_NAME:
          tf_config_job_name = constant.DEFAULT_TASK_NAME

      if tf_config_worker_hosts is None or tf_config_job_name is None \
          or tf_config_task_index is None:
        raise ValueError("Get hosts information failed for incomplete "
                         "TF_CONFIG: %s." % tf_config)

      worker_hosts = ",".join(tf_config_worker_hosts)
      if tf_config_ps_hosts:
        ps_hosts = ",".join(tf_config_ps_hosts)
      job_name = tf_config_job_name
      worker_index = tf_config_task_index

    if not ps_hosts:
      hosts = worker_hosts
      self._worker_index = worker_index
    else:  # support ps_hosts and process as worker_hosts
      hosts = worker_hosts + "," + ps_hosts
      worker_list = worker_hosts.split(",")
      worker_num = len(worker_list)
      if job_name == constant.DEFAULT_TASK_NAME:
        self._worker_index = worker_index
      else:
        self._worker_index = worker_index + worker_num
    self._hosts = hosts
    self._worker_num = len(hosts.split(","))
    self._worker_num_per_machine = 1
    self._cluster_spec = \
        server_lib.ClusterSpec({
            constant.DEFAULT_TASK_NAME: self._hosts.split(",")})
    self._available_devices = self.available_gpus()
    self.set_default_device()
    self._layout = None
    self._layout_type = layout
    self._virtual_devices = []
    if layout:
      self.generate_virtual_devices()
    tf_logging.info(self)

    # add metric
    if self._worker_index == 0:
      metric.add_metric(constant.DISTRIBUTED_FRAMEWORK,
                        constant.DISTRIBUTED_FRAMEWORK_NAME)

  def generate_virtual_devices(self, layout=None):
    """Generate virtual devices"""
    if self._virtual_devices:
      tf_logging.warn("Virtual devices are not empty, return existing ones.")
      return self._virtual_devices
    layout_type = layout if layout else self._layout_type
    self._layout = Layout(layout_type)
    self._layout.reorder_hosts(self)
    self._slices = self._layout.slice(self)
    local_device_indices = None
    for index, devices in enumerate(self._slices):
      vd = VirtualDevice(index, devices, self._worker_index, local_device_indices)
      self._virtual_devices.append(vd)
      if index == 0:
        local_device_indices = vd.local_device_indices
    return self._virtual_devices

  @property
  def virtual_devices(self):
    return self._virtual_devices

  def available_gpus(self):
    """Get available gpu list."""
    config = Env.get().get_config_proto()
    local_device_protos = device_lib.list_local_devices(config)
    count = len(
        [x.name for x in local_device_protos if x.device_type == "GPU"])
    devices = []
    for gpu_idx in range(count):
      devices.append(
          common.get_device_string(task=self._worker_index, device_index=gpu_idx))
    return devices

  def current_worker_chief_gpu(self):
    return common.get_device_string(task=self._worker_index)

  def current_worker_cpu(self):
    return common.get_device_string(task=self._worker_index, device_type="CPU")

  def set_default_device(self, tf_graph=None):
    """Set default device as GPU for tf graph."""
    if tf_graph is None:
      tf_graph = common.get_default_tf_graph()
    if not tf_graph._graph_device_function_stack._stack: # pylint: disable=protected-access
      tf_graph._add_device_to_stack(self.current_worker_chief_gpu()) # pylint: disable=protected-access

  def get_local_rank(self, global_rank):
    return global_rank % self.gpu_num_per_worker

  @property
  def worker_index(self):
    return self._worker_index

  @worker_index.setter
  def worker_index(self, worker_index):
    self._worker_index = worker_index

  @property
  def worker_num(self):
    return self._worker_num

  @property
  def gpu_num_per_worker(self):
    return len(self._available_devices)

  @property
  def available_devices(self):
    return self._available_devices

  @property
  def cluster_spec(self):
    return self._cluster_spec

  @property
  def total_gpu_num(self):
    return self.gpu_num_per_worker * self.worker_num

  @property
  def hosts(self):
    return self._hosts

  @hosts.setter
  def hosts(self, hosts):
    self._hosts = hosts

  @property
  def worker_num_per_machine(self):
    return self._worker_num_per_machine

  @worker_num_per_machine.setter
  def worker_num_per_machine(self, worker_num_per_machine):
    self._worker_num_per_machine = worker_num_per_machine

  def __str__(self):
    return "ClusterSpec: %s, WorkerNumber:%s, TaskIndex: %s, " \
           "AvailableDevices: %s, Layout:%s, VirtualDevices: %s" % (
               self._cluster_spec,
               self.worker_num,
               self._worker_index,
               self._available_devices,
               self._layout,
               self._virtual_devices)

  def __repr__(self):
    return self.__str__()

  def __enter__(self):
    Env.get().cluster = self

  def __exit__(self, unused_exception_type, unused_exception_value,
               unused_traceback):
    # Keep cluster information in Env.
    pass
