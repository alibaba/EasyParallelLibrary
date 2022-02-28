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
"""Test for hook of session run."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from distutils.version import LooseVersion as Version
import six
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework.versions import __version__

import epl
from epl.parallel.hooks import _append_replicated_fetches


# pylint: disable=missing-docstring,unused-argument,unused-variable
class RunHookTest(test.TestCase):
  def test_for_append_replicated_fetches(self):
    epl.init(config=epl.Config({"communication.gradients_reduce_method": "sum"}))
    with epl.Cluster(worker_hosts="127.0.0.1:8001", worker_index=0):
      with epl.replicate(device_count=1):
        num_x = np.random.randint(0, 10, (500, 20)).astype(dtype=np.float32)
        num_y = np.random.randint(0, 10, 500).astype(dtype=np.int64)
        dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
                                .batch(10).repeat(1)
        iterator = dataset.make_initializable_iterator()
        tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                             iterator.initializer)
        x, labels = iterator.get_next()

        logits = tf.layers.dense(x, 2)
        logits = tf.layers.dense(logits, 10)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)
        epl.add_to_collection(loss, epl.GraphKeys.GLOBAL_MEAN_OBJECTS)
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.001,
                                               momentum=0.9)
        train_op = optimizer.minimize(loss, global_step=global_step)
      tf.train.MonitoredTrainingSession()

      # Test for a single operation/tensor.
      fetches = loss
      replicas = []
      fetches = _append_replicated_fetches(fetches, replicas)
      replicas = [rep.name for rep in replicas]
      self.assertEqual(fetches.name, "EPL_PARALLEL_STRATEGY/truediv:0")
      self.assertListEqual(replicas, [
          "EPL_PARALLEL_STRATEGY/truediv_1:0",
          "EPL_PARALLEL_STRATEGY/truediv_2:0",
          "EPL_PARALLEL_STRATEGY/truediv_3:0"
      ])

      fetches = train_op
      replicas = []
      fetches = _append_replicated_fetches(fetches, replicas)
      replicas = [rep.name for rep in replicas]
      # Test for nvidia-tf(1.15.4) and deeprec(1.15.5).
      if Version(__version__) >= Version("1.15.4") and Version(__version__) < Version("2.0"):
        suffix = "/group_deps"
      else:
        suffix = ""
      self.assertEqual(fetches.name, "Momentum" + suffix)
      self.assertEqual(replicas, [
          "EPL_REPLICA_1/Momentum" + suffix, "EPL_REPLICA_2/Momentum" +
          suffix, "EPL_REPLICA_3/Momentum" + suffix
      ])

      # Test for list fetches.
      fetches = [loss, train_op]
      replicas = []
      fetches = _append_replicated_fetches(fetches, replicas)
      fetches = [fetch.name for fetch in fetches]
      replicas = [rep.name for rep in replicas]
      self.assertListEqual(
          fetches, ["EPL_PARALLEL_STRATEGY/truediv:0", "Momentum" + suffix])
      self.assertListEqual(replicas, [
          "EPL_PARALLEL_STRATEGY/truediv_1:0",
          "EPL_PARALLEL_STRATEGY/truediv_2:0",
          "EPL_PARALLEL_STRATEGY/truediv_3:0", "EPL_REPLICA_1/Momentum" +
          suffix, "EPL_REPLICA_2/Momentum" + suffix,
          "EPL_REPLICA_3/Momentum" + suffix
      ])

      # Test for type of dict.
      fetches = {"loss": loss, "train_op": train_op}
      replicas = []
      fetches = _append_replicated_fetches(fetches, replicas)
      replicas = [rep.name for rep in replicas]
      self.assertEqual(fetches["loss"].name,
                       "EPL_PARALLEL_STRATEGY/truediv:0")
      self.assertEqual(fetches["train_op"].name, "Momentum" + suffix)
      if six.PY2:
        self.assertListEqual(replicas, [
            "EPL_REPLICA_1/Momentum" + suffix, "EPL_REPLICA_2/Momentum" +
            suffix, "EPL_REPLICA_3/Momentum" + suffix,
            "EPL_PARALLEL_STRATEGY/truediv_1:0",
            "EPL_PARALLEL_STRATEGY/truediv_2:0",
            "EPL_PARALLEL_STRATEGY/truediv_3:0"
        ])
      else:
        self.assertListEqual(replicas, [
            "EPL_PARALLEL_STRATEGY/truediv_1:0",
            "EPL_PARALLEL_STRATEGY/truediv_2:0",
            "EPL_PARALLEL_STRATEGY/truediv_3:0", "EPL_REPLICA_1/Momentum" +
            suffix, "EPL_REPLICA_2/Momentum" + suffix,
            "EPL_REPLICA_3/Momentum" + suffix
        ])

      # Test for type of OrderedDict
      fetches = collections.OrderedDict()
      fetches["loss"] = loss
      fetches["train_op"] = train_op
      replicas = []
      fetches = _append_replicated_fetches(fetches, replicas)
      replicas = [rep.name for rep in replicas]
      self.assertEqual(fetches["loss"].name,
                       "EPL_PARALLEL_STRATEGY/truediv:0")
      self.assertEqual(fetches["train_op"].name, "Momentum" + suffix)
      self.assertListEqual(replicas, [
          "EPL_PARALLEL_STRATEGY/truediv_1:0",
          "EPL_PARALLEL_STRATEGY/truediv_2:0",
          "EPL_PARALLEL_STRATEGY/truediv_3:0", "EPL_REPLICA_1/Momentum" +
          suffix, "EPL_REPLICA_2/Momentum" + suffix,
          "EPL_REPLICA_3/Momentum" + suffix
      ])

      # Test for type of tuple.
      fetches = (loss, train_op)
      replicas = []
      fetches = _append_replicated_fetches(fetches, replicas)
      replicas = [rep.name for rep in replicas]
      self.assertEqual(fetches[0].name, "EPL_PARALLEL_STRATEGY/truediv:0")
      self.assertEqual(fetches[1].name, "Momentum" + suffix)
      self.assertListEqual(replicas, [
          "EPL_PARALLEL_STRATEGY/truediv_1:0",
          "EPL_PARALLEL_STRATEGY/truediv_2:0",
          "EPL_PARALLEL_STRATEGY/truediv_3:0", "EPL_REPLICA_1/Momentum" +
          suffix, "EPL_REPLICA_2/Momentum" + suffix,
          "EPL_REPLICA_3/Momentum" + suffix
      ])

      # Test for type of namedtuple.
      fetch_type = collections.namedtuple("fetch_type", ["loss", "train_op"])
      fetches = fetch_type(loss=loss, train_op=train_op)
      replicas = []
      fetches = _append_replicated_fetches(fetches, replicas)
      replicas = [rep.name for rep in replicas]
      self.assertEqual(fetches.loss.name, "EPL_PARALLEL_STRATEGY/truediv:0")
      self.assertEqual(fetches.train_op.name, "Momentum" + suffix)
      self.assertListEqual(replicas, [
          "EPL_PARALLEL_STRATEGY/truediv_1:0",
          "EPL_PARALLEL_STRATEGY/truediv_2:0",
          "EPL_PARALLEL_STRATEGY/truediv_3:0", "EPL_REPLICA_1/Momentum" +
          suffix, "EPL_REPLICA_2/Momentum" + suffix,
          "EPL_REPLICA_3/Momentum" + suffix
      ])

      # Test for nested list fetches.
      def _flatten(li):
        return sum(
            ([x] if not isinstance(x, list) else _flatten(x) for x in li), [])

      fetches = [labels, [train_op, logits, [loss, global_step]]]
      replicas = []
      fetches = _append_replicated_fetches(fetches, replicas)
      fetches = _flatten(fetches)
      fetches = [fetch.name for fetch in fetches]
      replicas = [rep.name for rep in replicas]
      self.assertListEqual(fetches, [
          "IteratorGetNext:1", "Momentum" + suffix, "dense_1/BiasAdd:0",
          "EPL_PARALLEL_STRATEGY/truediv:0", "global_step:0"
      ])
      self.assertListEqual(replicas, [
          "EPL_REPLICA_1/IteratorGetNext:1",
          "EPL_REPLICA_2/IteratorGetNext:1",
          "EPL_REPLICA_3/IteratorGetNext:1", "EPL_REPLICA_1/Momentum" +
          suffix, "EPL_REPLICA_2/Momentum" + suffix,
          "EPL_REPLICA_3/Momentum" + suffix,
          "EPL_REPLICA_1/dense_1/BiasAdd:0",
          "EPL_REPLICA_2/dense_1/BiasAdd:0",
          "EPL_REPLICA_3/dense_1/BiasAdd:0",
          "EPL_PARALLEL_STRATEGY/truediv_1:0",
          "EPL_PARALLEL_STRATEGY/truediv_2:0",
          "EPL_PARALLEL_STRATEGY/truediv_3:0",
          "EPL_REPLICA_1/global_step:0",
          "EPL_REPLICA_2/global_step:0",
          "EPL_REPLICA_3/global_step:0"
      ])

    # Test for nested list with dict.
    fetches = [labels, {"loss": loss}]
    replicas = []
    fetches = _append_replicated_fetches(fetches, replicas)
    replicas = [rep.name for rep in replicas]
    self.assertEqual(fetches[0].name, "IteratorGetNext:1")
    self.assertEqual(fetches[1]["loss"].name,
                     "EPL_PARALLEL_STRATEGY/truediv:0")
    self.assertListEqual(replicas, [
        "EPL_REPLICA_1/IteratorGetNext:1",
        "EPL_REPLICA_2/IteratorGetNext:1",
        "EPL_REPLICA_3/IteratorGetNext:1",
        "EPL_PARALLEL_STRATEGY/truediv_1:0",
        "EPL_PARALLEL_STRATEGY/truediv_2:0",
        "EPL_PARALLEL_STRATEGY/truediv_3:0"
    ])

    # Test for nested list with tuple.
    fetches = [labels, (loss, global_step)]
    replicas = []
    fetches = _append_replicated_fetches(fetches, replicas)
    replicas = [rep.name for rep in replicas]
    self.assertEqual(fetches[0].name, "IteratorGetNext:1")
    self.assertEqual(fetches[1][0].name, "EPL_PARALLEL_STRATEGY/truediv:0")
    self.assertEqual(fetches[1][1].name, "global_step:0")
    self.assertListEqual(replicas, [
        "EPL_REPLICA_1/IteratorGetNext:1",
        "EPL_REPLICA_2/IteratorGetNext:1",
        "EPL_REPLICA_3/IteratorGetNext:1",
        "EPL_PARALLEL_STRATEGY/truediv_1:0",
        "EPL_PARALLEL_STRATEGY/truediv_2:0",
        "EPL_PARALLEL_STRATEGY/truediv_3:0",
        "EPL_REPLICA_1/global_step:0",
        "EPL_REPLICA_2/global_step:0",
        "EPL_REPLICA_3/global_step:0"
    ])


# pylint: enable=missing-docstring,unused-argument,unused-variable

if __name__ == "__main__":
  test.main()
