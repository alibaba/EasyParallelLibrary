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
"""Test for header."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path
from tensorflow.python.platform import test


class HeaderTest(test.TestCase):
  """Test for header."""

  def _check_header(self, path):
    with open(str(path), "rb") as f:
      text = f.read().decode("UTF-8")
    self.assertTrue('copyright 2021 alibaba group' in text.lower(), "{} header is not ok.".format(path))

  def test_header(self):
    """Test python header."""
    suffix = ["*.py", "*.cc"]
    folders = ["./", "../epl", "../setup.py", "../csrc"]
    for folder in folders:
      for sf in suffix:
        for path in Path(folder).rglob(sf):
          self._check_header(path)


if __name__ == "__main__":
  test.main()
