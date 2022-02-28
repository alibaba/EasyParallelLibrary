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
"""Script to generate cflags."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

if hasattr(tf.sysconfig, "get_compile_flags"):
  cflags = tf.sysconfig.get_compile_flags()
else:
  tf_include = os.path.join(os.path.dirname(tf.__file__), "include")
  cflags = [
      '-I{}'.format(tf_include),
      '-I{}/external/nsync/public/'.format(tf_include)]

print(' '.join(cflags))
