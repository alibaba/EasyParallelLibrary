# 快速开始

我们将通过一个简单的模型示例来演示如何使用EPL来实现一个分布式训练程序。

## EPL 分布式策略表达

用户首先需要在本地模型定义文件`local_model.py`上添加分布式策略的定义。
下面这个例子展示了一个通过添加三行代码实现数据并行的例子。

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


## 启动分布式训练

定义好模型之后，用户需要提供一个本地单级单卡启动的训练脚本，比如`run.sh`.

```bash
# run.sh
python local_model.py
```

通过下面的脚本我们可以拉起一个单机两卡的数据并行训练任务。

```bash
epl-launch --num_workers 1 --gpu_per_worker 2 run.sh
```
