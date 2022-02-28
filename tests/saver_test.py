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
"""Test for tf.train.Saver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import shutil
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

import epl
from epl.ir.graph import Graph
from epl.runtime.saver import ShardingLoader, MemoryEfficientBuilder


# pylint: disable=missing-docstring,unused-argument,unused-variable,line-too-long,protected-access
class SaverTest(test.TestCase):
  def test_for_saver(self):
    epl.init()
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

        logits = tf.layers.dense(x, 10)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                      logits=logits)

      saver = tf.train.Saver()

      global_step = tf.train.get_or_create_global_step()
      optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
      train_op = optimizer.minimize(loss, global_step=global_step)
      tf.train.MonitoredTrainingSession()

      g = Graph.get()
      # Check forward operations
      for op in g.taskgraphs[0].operations.forward_operations(0, 0):
        self.assertFalse(op.name.startswith("save/"))
      for op in g.taskgraphs[0].operations.backward_operations(0, 0):
        self.assertFalse(op.name.startswith("save/"))
      for op in g.taskgraphs[0].operations.apply_operations(0):
        self.assertFalse(op.name.startswith("save/"))

  def _model_def(self, optimizer=None, repeat=1, size=16):
    num_x = np.random.randint(0, 10, (500, 10)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
        .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    x, _ = iterator.get_next()
    x = tf.layers.dense(inputs=x, units=16, activation=None)
    for i in range(repeat):
      x = tf.layers.dense(inputs=x, units=16, activation=None,
                          name="layer_{}".format(i))
    dense1 = tf.layers.dense(inputs=x, units=size, activation=None,
                             name='split1')
    logits = tf.layers.dense(inputs=dense1, units=10, activation=None,
                             name='split0')
    loss = tf.reduce_mean(logits)
    global_step = tf.train.get_or_create_global_step()
    opt_name = "momentum"
    if optimizer == 'adam':
      opt_name = "adam"
      with tf.name_scope(opt_name):
        optimizer = tf.train.AdamOptimizer(learning_rate=0)
    else:
      with tf.name_scope(opt_name):
        optimizer = tf.train.MomentumOptimizer(learning_rate=0, momentum=0.09)
    with tf.name_scope(opt_name):
      train_op = optimizer.minimize(loss, global_step=global_step)
    return [loss, train_op, global_step]

  def load_case(self, config, ckpt_dir=None, restore=None, optimizer=None,
                repeat=1, func=None, size=16):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    res = []
    max_steps = 5
    with tf.Graph().as_default():
      train_opts = self._model_def(optimizer, repeat, size)
      if restore is not None:
        loader = restore(ckpt_dir)
        loader.restore()
        ckpt_dir = None
      with tf.train.MonitoredTrainingSession(checkpoint_dir=ckpt_dir) as sess:
        graph = tf.get_default_graph()
        name2dev = {o.name: o.outputs[0] for o in graph.get_operations() if
                    o.type == 'VariableV2'}
        train_opts.append(name2dev)
        for i in range(max_steps):
          train_loss, _, step, value = sess.run(train_opts)
          res.append((train_loss, step, value))
    return res, name2dev

  def test_save_load_ckpt_normal(self):
    def get_loader(ckpt_dir):
      latest_ckp = tf.train.latest_checkpoint(ckpt_dir)
      loader = ShardingLoader(latest_ckp)
      return loader
    tempfolder = tempfile.mkdtemp()
    config = epl.Config()
    res1, name2dev = self.load_case(config, tempfolder)
    step1 = [r[1] for r in res1]
    self.assertEqual(step1, [0, 1, 2, 3, 4])
    config = epl.Config()
    res0, name2dev = self.load_case(config, tempfolder, get_loader)
    shutil.rmtree(tempfolder)
    step0 = [r[1] for r in res0]
    self.assertEqual(step0, [5, 6, 7, 8, 9])

  def test_save_load_ckpt_varlist(self):
    def get_loader(ckpt_dir):
      var_list = variables._all_saveable_objects()
      var_list = [v for v in var_list if "adam" not in v.name.lower()]
      latest_ckp = tf.train.latest_checkpoint(ckpt_dir)
      loader = ShardingLoader(latest_ckp, var_list)
      return loader
    tempfolder = tempfile.mkdtemp()
    config = epl.Config()
    res1, name2dev = self.load_case(config, tempfolder)
    step1 = [r[1] for r in res1]
    self.assertEqual(step1, [0, 1, 2, 3, 4])
    config = epl.Config()
    var_list = variables._all_saveable_objects()
    var_list = [v for v in var_list if "" not in var_list]
    res0, name2dev = self.load_case(config, tempfolder, get_loader,
                                    optimizer="adam")
    shutil.rmtree(tempfolder)
    step0 = [r[1] for r in res0]
    self.assertEqual(step0, [5, 6, 7, 8, 9])

  def test_save_load_ckpt_repeat(self):
    def get_loader(ckpt_dir):
      latest_ckp = tf.train.latest_checkpoint(ckpt_dir)
      var_list = variables._all_saveable_objects()
      var_list = [v for v in var_list if 'global_step' not in v.name]
      assign_map = {}
      for v in var_list:
        name = re.sub(r'layer_\d+/', 'layer_0/', v.op.name)
        assign_map[v.op.name] = name
      loader = ShardingLoader(latest_ckp, var_list=var_list,
                              assign_map=assign_map)
      return loader
    tempfolder = tempfile.mkdtemp()
    config = epl.Config()
    res1, name2dev0 = self.load_case(config, tempfolder, repeat=1)
    step1 = [r[1] for r in res1]
    self.assertEqual(step1, [0, 1, 2, 3, 4])
    config = epl.Config()
    var_list = variables._all_saveable_objects()
    var_list = [v for v in var_list if "" not in var_list]
    res0, name2dev1 = self.load_case(config, tempfolder, get_loader, repeat=5)

    step0 = [r[1] for r in res0]
    self.assertEqual(step0, [0, 1, 2, 3, 4])
    reader = tf.train.load_checkpoint(tempfolder)

    shape_from_key = reader.get_variable_to_shape_map()
    weights1 = {key: reader.get_tensor(key) for key in shape_from_key}
    weights0 = [r[2] for r in res0][0]
    for w in weights0:
      name = re.sub(r'layer_\d+/', 'layer_0/', w)
      if 'global_step' in name: continue
      self.assertTrue((weights0[w] == weights1[name]).all())
    shutil.rmtree(tempfolder)

  def test_save_load_ckpt_shard(self):
    def get_loader(ckpt_dir):
      latest_ckp = tf.train.latest_checkpoint(ckpt_dir)
      var_list = variables._all_saveable_objects()
      var_list = [v for v in var_list if 'global_step' not in v.name]
      assign_map = {}
      sharding_info = {}
      for v in var_list:
        name = re.sub(r'layer_\d+/', 'layer_0/', v.op.name)
        assign_map[v.op.name] = name
        if v.name.startswith("split"):
          shape = v.shape.as_list()
          if len(shape) == 2:
            sharding_info[name] = {"begin": [0, 0], "size": shape}
          else:
            sharding_info[name] = {"begin": [0], "size": shape}

      loader = ShardingLoader(latest_ckp, var_list=var_list,
                              assign_map=assign_map,
                              sharding_info=sharding_info)
      return loader
    tempfolder = tempfile.mkdtemp()
    config = epl.Config()
    res1, name2dev0 = self.load_case(config, tempfolder, repeat=1)
    step1 = [r[1] for r in res1]
    self.assertEqual(step1, [0, 1, 2, 3, 4])
    config = epl.Config()
    var_list = variables._all_saveable_objects()
    var_list = [v for v in var_list if "" not in var_list]
    res0, name2dev1 = self.load_case(config, tempfolder, get_loader, repeat=5,
                                     size=8)

    step0 = [r[1] for r in res0]
    self.assertEqual(step0, [0, 1, 2, 3, 4])
    reader = tf.train.load_checkpoint(tempfolder)

    shape_from_key = reader.get_variable_to_shape_map()
    weights1 = {key: reader.get_tensor(key) for key in shape_from_key}
    weights0 = [r[2] for r in res0][0]
    for w in weights0:
      name = re.sub(r'layer_\d+/', 'layer_0/', w)
      if 'global_step' in name: continue
      w1 = weights1[name]
      w0 = weights0[w]
      if 'split' in w:
        if len(w1.shape) == 2:
          w1 = w1[0:w0.shape[0], 0:w0.shape[1]]
        else:
          w1 = w1[0:w0.shape[0]]
      self.assertTrue((w0 == w1).all())
    shutil.rmtree(tempfolder)

  def memory_saver(self, tempfolder):
    epl.init()
    epl.set_default_strategy(epl.replicate(1))
    num_x = np.random.randint(0, 10, (500, 20)).astype(dtype=np.float32)
    num_y = np.random.randint(0, 10, 500).astype(dtype=np.int64)
    dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)) \
                             .batch(10).repeat(1)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
                         iterator.initializer)
    x, labels = iterator.get_next()

    logits = tf.layers.dense(x, 10)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits)
    global_step = tf.train.get_or_create_global_step()
    optimizer = tf.train.MomentumOptimizer(learning_rate=0.00, momentum=0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    saver = tf.train.Saver(sharded=True, restore_sequentially=True, \
                           builder=MemoryEfficientBuilder())
    ckpt_file = tf.train.latest_checkpoint(tempfolder)

    loader = ShardingLoader(ckpt_file)
    loader.restore()
    def get_session(sess):
      session = sess
      while type(session).__name__ != 'Session':
        # pylint: disable=W0212
        session = session._sess
      return session
    with tf.train.MonitoredTrainingSession() as sess:
      if ckpt_file is None:
        for i in range(5):
          _, step, loss0 = sess.run([train_op, global_step, loss])
          print(step, loss0)
        saver.save(get_session(sess),
                   os.path.join(tempfolder, "model.ckpt"),
                   step)
      values = sess.run([{v.name: v for v in variables._all_saveable_objects()}])
    return values[0], ckpt_file

  def test_memory_saver(self):
    tempfolder = tempfile.mkdtemp()
    res1, ckpt1 = self.memory_saver(tempfolder)
    res2, ckpt2 = self.memory_saver(tempfolder)
    self.assertEqual(ckpt1, None)
    self.assertTrue(ckpt2 is not None)
    for name, arr in res1.items():
      arr2 = res2[name]
      self.assertTrue((arr == arr2).all())
    shutil.rmtree(tempfolder)


# pylint: enable=missing-docstring,unused-argument,unused-variable,line-too-long
# pyline: enable=protected-access
if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  test.main()
