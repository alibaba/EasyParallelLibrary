## 接口使用范例

本文档主要介绍如何使用EPL的[并行化接口](strategy.md)实现常见的并行化策略，以及复杂的混合并行。

### 数据并行

```python
import epl
epl.init()
with epl.replicate(device_count=1):
  model()
```

上面这个例子是一个数据并行的例子，每个模型副本用一张卡来计算。如果用户申请了8张卡，就是一个并行度为8的数据并行任务。

### 流水并行

```python
import epl

config = epl.Config({"pipeline.num_micro_batch": 4})
epl.init(config)
with epl.replicate(device_count=1, name="stage_0"):
  model_part1()
with epl.replicate(device_count=1, name="stage_1"):
  model_part2()
```
在上述例子中，模型被切分成2个 `TaskGraph` "stage_0"和"stage_1"，用户可以通过配置`pipeline.num_micro_batch`参数来设定Pipeline的micro batch数量。
在这个例子里，"stage_0"和"stage_1"组成一个模型副本，共需要2张GPU卡。如果用户申请了8张卡，EPL会自动在pipeline外嵌套一层并行度为4的数据并行（4个pipeline副本并行执行）。

### 算子拆分
#### 算子拆分 - 大规模图像分类

```python
import epl
config = epl.Config({"cluster.colocate_split_and_replicate": True})
epl.init(config)
with epl.replicate(8):
  resnet()
with epl.split(8):
  classification()
```
上述是一个大规模图像分类的例子，在这个例子中，对图像特征部分采用数据并行，对分类层采用算子拆分的方式。
为了减少两个TaskGraph直接的通信开销，我们可以通过设置`cluster.colocate_split_and_replicate`参数将两个TaskGraph放置在相同的卡上（默认不同的TaskGraph会放置在不同的卡上）。

#### 算子拆分 - MOE Transformer

```
import epl
config = epl.Config({"cluster.colocate_split_and_replicate": True})
epl.init(config)
total_gpu_num = epl.Env.get().cluster.total_gpu_num

epl.set_default_strategy(epl.replicate(total_gpu_num))

AttentionAndGating()

with epl.split(total_gpu_num):
  MOE_Variable_Define()

MOE_Calculation_Define()
```
在上述例子中，我们实现了一个简单的MOE模型，通过设置`set_default_strategy` 设置默认的并行化策略为`replicate`,
并对MOE部分进行计算的拆分。
