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
"""Pai metrics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import tf_logging

def add_metric(key, value):
  """Add metric."""
  try:
    from pai_metrics import MetricsManager

    if MetricsManager.metrics_enabled():
      mm = MetricsManager()
      mm.add_metric_async(key, value)
    else:
      tf_logging.warn("Not enabled pai metrics, ignore metric key: {}, "
                      "value: {}".format(key, value))
  except ImportError:
    tf_logging.warn("Not found pai metrics, ignore metric key: {}, "
                    "value: {}".format(key, value))
