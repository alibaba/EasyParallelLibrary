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
"""Simple DNN algorithm of data parallelism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import epl

tf.logging.set_verbosity(tf.logging.INFO)
flags = tf.app.flags
flags.DEFINE_integer("max_steps", 10, "max training step")
flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
flags.DEFINE_integer("num_micro_batch", 1, "num_micro_batch")
flags.DEFINE_string("amp", None, "amp")
FLAGS = tf.app.flags.FLAGS
config_json = {}
if FLAGS.amp:
  config_json["amp.level"] = "o1"
  config_json["amp.loss_scale"] = float(FLAGS.amp) if FLAGS.amp != "dynamic" else "dynamic"
  config_json["pipeline.num_micro_batch"] = FLAGS.num_micro_batch
epl.init(epl.Config(config_json))
epl.set_default_strategy(epl.replicate(1))

# dataset
num_x = np.random.randint(0, 10, (500, 20)).astype(dtype=np.float32)
num_y = np.random.randint(0, 10, 500).astype(dtype=np.int64)
dataset = tf.data.Dataset.from_tensor_slices((num_x, num_y)).batch(10).repeat(1)
iterator = dataset.make_initializable_iterator()
tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
x, labels = iterator.get_next()

logits = tf.layers.dense(x, 2)
logits = tf.layers.dense(logits, 10)
loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

global_step = tf.train.get_or_create_global_step()
optimizer = tf.train.MomentumOptimizer(FLAGS.learning_rate, momentum=0.9)
train_op = optimizer.minimize(loss, global_step=global_step)

hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps)]
max_steps = (FLAGS.max_steps+1) * FLAGS.num_micro_batch
cum_steps = 0
with tf.train.MonitoredTrainingSession(hooks=hooks) as sess:
  while not sess.should_stop() or cum_steps > max_steps:
    train_ops = [loss, train_op, global_step]
    if FLAGS.amp:
      train_ops.append(epl.Env.get().parallel_information["AMP_LOSS_SCALE"]._num_good_steps) # pylint: disable=protected-access
      train_ops.append(epl.Env.get().parallel_information["AMP_LOSS_SCALE"]._current_loss_scale) # pylint: disable=protected-access
    res = sess.run(train_ops)
    print("Iteration %s , Loss: %s ." % (res[2], res[0]))
    if FLAGS.amp:
      num_good_steps = res[3]
      current_loss_scale = res[4]
      if FLAGS.learning_rate >= 100:
        assert num_good_steps <= 1
        assert res[2] <= 1
      print('good_steps: {}, current_loss_scale: {}'.format(num_good_steps, current_loss_scale))
    cum_steps += 1
print("Train Finished.")
