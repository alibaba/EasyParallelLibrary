# Quick Start

In this section, we will use a simple DNN training example to show
how to use EPL for distributed training.

## EPL Annotation

A user needs to first annotate `local_model.py` with EPL parallelism
strategies. The following example shows a data parallelism sample by adding
three lines.

```diff
# local_model.py
import numpy as np
import tensorflow as tf
+ import epl

+ epl.init()
+ epl.set_default_strategy(epl.replicate(1))

# Define model
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
optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
train_op = optimizer.minimize(loss, global_step=global_step)

# Training session
with tf.train.MonitoredTrainingSession() as sess:
  for i in range(10):
    train_loss, _, step = sess.run([loss, train_op, global_step])
    print("Iteration %s , Loss: %s ." % (step, train_loss))
print("Train Finished.")
```


## Launch a parallel training

Then the user needs to provide a local launch script such as `run.sh`, as follows:

```bash
# run.sh
python local_model.py
```
The following script launches a parallel training program with 1 worker and 2 GPUs.

```bash
epl-launch --num_workers 1 --gpu_per_worker 2 run.sh
```
