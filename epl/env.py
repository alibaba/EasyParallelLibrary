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
"""Classes for saving epl runtime information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import server_lib

from epl.utils import common, constant
from epl.communicators.collective_keys import CollectiveKeys
from epl.config import Config
from epl.utils.version import VERSION
from epl.strategies.strategy_context import StrategyContext
DEFAULT_ENV = None

_ENV = threading.local()


class Env(object):
  """Env for saving runtime context and confiuration information."""
  def __init__(self):
    self._init()

  @classmethod
  def get(cls):
    """Get static env."""

    global DEFAULT_ENV
    if DEFAULT_ENV:
      return DEFAULT_ENV
    DEFAULT_ENV = Env()
    return DEFAULT_ENV

  def _init(self):
    """Init env."""
    self._cluster = None
    self._server = None
    self._config = None
    self._collective_keys = None
    self._is_ready = False
    self._strategy_context = None
    self._graph_map = {}
    # Keep epl graphs in create order.
    self._epl_graphs = []
    self._parallel_map = {}

  def reset(self):
    for _, graph in self._graph_map.items():
      graph.reset()
    self._graph_map.clear()
    self._epl_graphs = []
    self._parallel_map.clear()
    self._init()

  @property
  def collective_keys(self):
    if not self._collective_keys:
      self._collective_keys = [CollectiveKeys() for _ in range(self.cluster.gpu_num_per_worker)]
    return self._collective_keys

  @property
  def server(self):
    return self._server

  @server.setter
  def server(self, server):
    if self._server is not None:
      raise RuntimeError("Server is already set.")
    self._server = server

  @property
  def cluster(self):
    return self._cluster

  @cluster.setter
  def cluster(self, cluster):
    """Set cluster."""
    self._cluster = cluster

  @property
  def strategy_context(self):
    return self._strategy_context

  @property
  def is_ready(self):
    return self._is_ready

  @property
  def config(self):
    return self._config

  def init(self, config=None):
    """Initialize epl Env."""
    from epl.parallel import hooks

    # NOTE: Setting 'PAITF_TRACING_CONCURRENT_KERNEL' environ to enable
    # concurrent kernel launching on GPU device during profiling session steps.
    # This is neccesary for NCCL concurrent communication in PAISoar. Otherwise
    # it would be possible of getting hanged during NCCL communication phase in
    # the profiling steps.
    os.environ["PAITF_TRACING_CONCURRENT_KERNEL"] = "True"

    self._config = config if config else Config()
    self._strategy_context = StrategyContext()
    hooks.add_hooks()
    self._is_ready = True
    tf_logging.info("EPL is enabled. EPL version: %s ." % VERSION)
    tf_logging.info("Using EPL Config: \n {}".format(config))

  def get_config_proto(self, allow_growth=True):
    """Return ConfigProto, default gpu_options.allow_growth = True."""
    config = config_pb2.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    # Try to get visible devices from ENV.
    if self.config:
      visible_devices = self.config.cluster.run_visible_devices
      if visible_devices:
        config.gpu_options.visible_device_list = visible_devices
    return config

  @property
  def default_graph(self):
    """Get default epl graph."""
    tf_graph = common.get_default_tf_graph()
    epl_graph = None
    if tf_graph in self._graph_map:
      epl_graph = self._graph_map[tf_graph]
    elif self._epl_graphs:
      epl_graph = self._epl_graphs[-1]
    return epl_graph

  @property
  def parallel_information(self):
    """Get parallel information from default epl graph."""
    info = None
    if self.default_graph:
      info = self.default_graph.parallel_information
    return info

  @property
  def graph_map(self):
    return self._graph_map

  @property
  def parallel_map(self):
    return self._parallel_map

  @property
  def epl_graphs(self):
    return self._epl_graphs

  def get_or_create_server(self):
    """Create tensorflow server."""
    if not self.cluster:
      raise RuntimeError("Create tensorflow server failed because of none cluster.")
    if not self.server:
      tf_server_obj = server_lib.Server
      config = self.get_config_proto()
      config.experimental.collective_group_leader = constant.DEFAULT_GROUP_LEADER
      self.server = tf_server_obj(self.cluster.cluster_spec,
                                  job_name=constant.DEFAULT_TASK_NAME,
                                  task_index=self.cluster.worker_index,
                                  config=config)
    return self.server
