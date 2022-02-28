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
"""Setup script."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import glob
import os
import subprocess
import sys
import shutil
from distutils.cmd import Command as DistutilsCommand
from distutils.command.build import build as DistutilsBuild

from setuptools import find_packages
from setuptools import setup
from setuptools.dist import Distribution



if sys.version_info[0] < 3:
  import imp
  VERSION = imp.load_source('epl.version', 'python/epl/utils/version.py').VERSION
else:
  from importlib.machinery import SourceFileLoader
  VERSION = SourceFileLoader("epl.version", "python/epl/utils/version.py") \
      .load_module().VERSION

PACKAGES = find_packages("python", exclude=['.*test'])
PACKAGE_DATA = {'': ['*.so']}
cwd = os.path.dirname(os.path.abspath(__file__))
cc_path = os.path.join(cwd, "cc")


class BinaryDistribution(Distribution):
  """This class is needed in order to create OS specific wheels."""
  def has_ext_modules(self):
    return True

def build_clean():
  """build clean."""
  subprocess.check_call(["make", "clean"], cwd=cc_path)
  patterns_to_remove = ["dist/", "build/", "*.egg-info/", "*.pyc", "*.orig", "events.out.tfevents.*"]
  for pattern in patterns_to_remove:
    for item in glob.glob(pattern):
      shutil.rmtree(item)
      print("remove {}".format(item))

class EplBuild(DistutilsBuild):
  """EPL build."""
  def run(self):
    build_clean()
    subprocess.check_call(["make", "-j8"], cwd=cc_path)
    DistutilsBuild.run(self)

class EplClean(DistutilsCommand):
  """EPL clean."""
  user_options = []
  def initialize_options(self):
    pass

  def finalize_options(self):
    pass

  def run(self):
    build_clean()

setup(
    name='epl',
    version=VERSION,
    packages=PACKAGES,
    include_package_data=True,
    package_data=PACKAGE_DATA,
    package_dir={"": "python"},
    distclass=BinaryDistribution,
    entry_points={
        "console_scripts": [
            "epl-launch=epl.utils.launcher:main",
        ]
    },
    cmdclass={'build': EplBuild, 'clean': EplClean},
    zip_safe=False,
    author='Alibaba Inc.',
    url='https://yuque.antfin-inc.com/pai-user/manual/intro',
    description=('Easy Parallel Library(EPL) powered by Alibaba.'),
    keywords=['distributed training', 'machine learning', 'tensorflow'],
    extras_require={
        ':python_version == "2.7"': [
            'pandas==0.24.2',
            'matplotlib==2.0.0'
        ],
        ':python_version >= "3.4"': [
            'pandas==1.1.5',
            'matplotlib==3.3.4',
            'toposort'
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
    license='Apache 2.0',
)
