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
"""EPL: Easy Parallel Library for efficient and large-scale training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.eager.context import executing_eagerly

from epl.cluster import Cluster
from epl.env import Env
from epl.config import Config

from epl.ir.graph import add_to_collection
from epl.ir.graph import get_all_collections
from epl.ir.graph import get_collection
from epl.ir.graph import Graph
from epl.ir.graph import GraphKeys
from epl.strategies.replicate import replicate
from epl.strategies.split import split

from epl.utils.version import VERSION


def init(config=None):
  """Init EPL."""
  if executing_eagerly():
    raise RuntimeError("Tensorflow eager mode is not supported by EPL now, " + \
                       "please do not call tf.enable_eager_execution()")
  env = Env.get()
  env.reset()
  env.init(config)
  # Create cluster with global device information.
  if env.config.cluster.colocate_split_and_replicate:
    env.cluster = Cluster()
  else:
    env.cluster = Cluster(layout=None)


def set_default_strategy(strategy):
  """Set default strategy."""
  Graph.get(may_create=True).set_default_strategy(strategy)
