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
"""Options for configuring different frameworks in distributed training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import os
from six.moves import xrange  # pylint: disable=redefined-builtin
from six import string_types as string

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import device_lib
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.platform import tf_logging as logging
try:
  from tensorflow.python.training import device_util
except: # pylint: disable=bare-except
  from tensorflow.python.distribute import device_util
from tensorflow.python.training import server_lib

from epl.communicators.base import Communicator
from epl.communicators.nccl import NcclCommunicator

class Options(object): # pylint: disable=useless-object-inheritance
  """Options for configuring different frameworks."""
  def __init__(self, user_kwargs, **default_kwargs):
    self.__items__ = dict(default_kwargs)
    self.__items__.update(user_kwargs)

  def __getattr__(self, attr):
    if attr not in self.__items__:
      raise AttributeError(attr)
    return self.__items__[attr]

  def __str__(self):
    return str(self.__items__)

  def update(self, **kwargs):
    self.__items__.update(kwargs)

class ServerSpec(object): # pylint: disable=useless-object-inheritance
  """Specification of cluster and current server."""

  _instance = None

  @classmethod
  def get(cls):
    """Get singleton."""
    if cls._instance is None:
      cls._instance = cls()
    return cls._instance

  def __init__(self):
    """Construct a server specification."""
    self._task_type = 'localhost'
    self._task_id = 0
    self._cluster_spec = None
    self._is_chief = True
    self._num_gpus = 0
    self._input_pipeline_id = 0
    self._update()

  def __str__(self):
    return 'ServerSpec {{{}:{} {} {}GPU, local={}, all={})}}'.format(
        self._task_type,
        self._task_id,
        'chief' if self._is_chief else '',
        self._num_gpus,
        self._local_devices,
        self._devices)

  @property
  def cluster_spec(self):
    """cluster spec."""
    return self._cluster_spec

  @property
  def task_type(self):
    """job name of current server. `localhost` by default."""
    return self._task_type

  @property
  def task_id(self):
    """task index of current server. 0 by default."""
    return self._task_id

  @property
  def is_chief(self):
    """True if current server is chief worker."""
    return self._is_chief

  @property
  def has_gpu(self):
    """True if current server has GPU."""
    return self._num_gpus > 0

  @property
  def default_device(self):
    """default device of current server."""
    return self._default_device

  @property
  def local_devices(self):
    """devices of current server."""
    return self._local_devices

  @property
  def devices(self):
    """devices of all servers."""
    return self._devices

  @property
  def local_cpu_device(self):
    """CPU0 device of current server."""
    return self._local_cpu_device

  @property
  def cpu_devices(self):
    """CPU devices of all servers."""
    return self._cpu_devices

  @property
  def input_pipeline_id(self):
    """Input pipeline ID."""
    return self._input_pipeline_id

  def device_index(self, device_or_tower_id):
    """Get global index of device or tower_id."""
    if isinstance(device_or_tower_id, string):
      return self._devices.index(device_or_tower_id)

    if self._num_gpus == 0:
      if device_or_tower_id > 0:
        raise ValueError('Only 1 tower for CPU-only worker')
      return self._task_id

    if device_or_tower_id >= self._num_gpus:
      raise ValueError(
          'Tower {} does not exist in the worker with {} towers'.format(
              device_or_tower_id, self._num_gpus))
    return self._num_gpus * self._task_id + int(device_or_tower_id)

  def current_index(self):
    """Get global index of current device."""
    return self._devices.index(Communicator.current_device())

  def _update(self, task_type=None, task_id=None, cluster_spec=None,
              num_gpus=0):
    """Update parameters from cluster_spec.

    If task_type, task_id or cluster_spec is None, these arguments will not be
    changed.

    Args:
      task_type: (Optional.) name of current job. `localhost` by default.
      task_id: (Optional.) index of current task. 0 by default.
      cluster_spec: (Optional.) ClusterSpec object.
    """
    tf_config = get_tf_config()
    if tf_config:
      self._task_type = tf_config.task_type
      self._task_id = tf_config.task_id
      self._cluster_spec = server_lib.ClusterSpec(tf_config.cluster)
    if task_type:
      self._task_type = task_type
    if task_id:
      self._task_id = task_id
    if cluster_spec:
      self._cluster_spec = cluster_spec
    if self._cluster_spec:
      self._cluster_spec = multi_worker_util.normalize_cluster_spec(
          self._cluster_spec)
    if self._cluster_spec:
      self._is_chief = multi_worker_util.is_chief(
          self._cluster_spec, self._task_type, self._task_id)
      if hasattr(multi_worker_util, 'id_in_cluster'):
        self._input_pipeline_id = multi_worker_util.id_in_cluster(
            self._cluster_spec, self._task_type, self._task_id)
    if num_gpus is None:
      num_gpus = 0
      num_gpus_config = config_pb2.ConfigProto()
      num_gpus_config.inter_op_parallelism_threads = 1
      num_gpus_config.intra_op_parallelism_threads = 1
      num_gpus_config.gpu_options.allow_growth = True
      for device in device_lib.list_local_devices(num_gpus_config):
        if device.device_type == 'GPU':
          num_gpus += 1
    self._num_gpus = num_gpus
    self._default_device = '/job:{}/replica:0/task:{}'.format(
        self._task_type, self._task_id)
    self._local_cpu_device = \
        device_util.canonicalize('/device:CPU:0', default=self._default_device)
    if self._num_gpus == 0:
      self._local_devices = [self._local_cpu_device]
    else:
      self._local_devices = [
          device_util.canonicalize(
              "/device:GPU:{}".format(d), default=self._default_device) \
              for d in xrange(self._num_gpus)]
    if not self._cluster_spec:
      self._devices = list(self._local_devices)
      return
    task_defs = dict(enumerate(self._cluster_spec.job_tasks(self._task_type)))
    task_indices = sorted(task_defs, key=task_defs.__getitem__)
    worker_indices = []
    try:
      worker_defs = dict(enumerate(self._cluster_spec.job_tasks('worker')))
      worker_indices = sorted(worker_defs, key=worker_defs.__getitem__)
    except: # pylint: disable=bare-except
      pass
    chief_indices = []
    try:
      chief_defs = dict(enumerate(self._cluster_spec.job_tasks('chief')))
      chief_indices = sorted(chief_defs, key=chief_defs.__getitem__)
    except: # pylint: disable=bare-except
      pass
    self._cpu_devices = [
        device_util.resolve(
            '/job:{}/task:{}/device:CPU:0'.format(self._task_type, t)) \
            for t in task_indices]
    if self._num_gpus == 0:
      self._devices = self._cpu_devices
      if self._task_type == 'worker':
        self._devices = [
            device_util.resolve(
                '/job:{}/task:{}/device:CPU:0'.format('chief', t)) \
                for t in chief_indices] + self._devices
      elif self._task_type == 'chief':
        self._devices += [
            device_util.resolve(
                '/job:{}/task:{}/device:CPU:0'.format('worker', t)) \
                for t in worker_indices]
      return
    self._devices = [
        device_util.resolve(
            '/job:{}/task:{}/device:GPU:{}'.format(self._task_type, t, g)) \
            for t in task_indices for g in xrange(self._num_gpus)]
    if self._task_type == 'worker':
      self._devices = [
          device_util.resolve(
              '/job:{}/task:{}/device:GPU:{}'.format('chief', t, g)) \
              for t in chief_indices for g in xrange(self._num_gpus)] + \
              self._devices
    elif self._task_type == 'chief':
      self._devices += [
          device_util.resolve(
              '/job:{}/task:{}/device:GPU:{}'.format('worker', t, g)) \
              for t in worker_indices for g in xrange(self._num_gpus)]

  def update(self, task_type=None, task_id=None, cluster_spec=None,
             num_gpus=0):
    """Update parameters from cluster_spec.

    If task_type, task_id or cluster_spec is None, these arguments will not be
    changed.

    Args:
      task_type: (Optional.) name of current job. `localhost` by default.
      task_id: (Optional.) index of current task. 0 by default.
      cluster_spec: (Optional.) ClusterSpec object.
    """
    self._update(
        task_type=task_type, task_id=task_id, cluster_spec=cluster_spec,
        num_gpus=num_gpus)
    logging.info('ServerSpec updated: {}'.format(self))

class CommunicatorSpec(object): # pylint: disable=useless-object-inheritance
  """Builder of communicators."""
  def __init__(self, devices=None, comm_impl=None, **kwargs):
    if not devices:
      devices = ServerSpec.get().devices
    self._devices = [
        device_util.canonicalize(d, default=Communicator.DEFAULT_DEVICE) \
        for d in devices]
    if not comm_impl:
      comm_impl = NcclCommunicator
    self._comm_impl = comm_impl
    self._comm_kwargs = kwargs or {}

  @property
  def devices(self):
    return self._devices

  @property
  def impl(self):
    return self._comm_impl

  @property
  def kwargs(self):
    return self._comm_kwargs

def build_communicator(shared_name, comm_spec):
  return Communicator.create(
      shared_name, comm_spec.devices, comm_spec.impl, **comm_spec.kwargs)

def get_tf_config():
  """Get configuration from TF_CONFIG environment variable.
  """
  tf_config = json.loads(os.getenv('TF_CONFIG', '{}'))
  if not tf_config:
    return None
  task = tf_config['task']
  cluster = tf_config['cluster']
  task_type = task['type']
  task_id = int(task['index'])
  tf_config_type = collections.namedtuple(
      "TfConfig", ["task_type", "task_id", "cluster"])
  return tf_config_type(task_type, task_id, cluster)
