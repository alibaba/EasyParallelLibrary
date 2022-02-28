# 配置

用户可以通过配置项开启各种优化功能。目前可以通过环境变量或者配置接口的方式来更改默认配置。

以下配置表格包含

- Param Key: 参数名, 在EPL中，参数名的命名规则为"param_category.attribute", `param_category`为参数的分类，比如`pipeline`, `attribute`为每个参数类别下的配置属性，比如`num_micro_batch`。
- Type: 参数类型
- Default: 默认值
- Description: 解释

接口定义：
```python
Config(param_dict=None)
```

| Args | Type| Required | Description |
|:----:|:----:|:---:|:-----------:|
| param_dict | dict | False | 参数字典，key为参数名，value为参数值。 |

示例：

```python
import epl
config = epl.Config({"pipeline.num_micro_batch": 4})
epl.init(config)
```
在上述例子中用户通过构造一个dict类型的参数字典，来修改参数配置。具体的参数描述请查阅下文的参数列表。


## Pipeline 配置
|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "pipeline.num_micro_batch"    |  integer    |   1  |   Pipeline number of micro batches.  | 
|     "pipeline.num_stages"    |  integer    |     None  |     如果开启了自动Pipeline, 可以配置pipeline的stage数。 | 
|     "pipeline.strategy"   |  str    |   "PreferBackward"  |     Pipeline调度策略，可选策略为 ["PreferBackward", "PreferForward"] | 

- `PreferBackward`: 优先后向计算的调度策略，类似 [PipeDream](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/).
- `PreferForward`: 优先前向计算的调度策略，类似 [GPipe](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html).


## Gradient Checkpoint (GC) 配置

Gradient checkpoint通过重算换显存的方式来降低训练过程中的峰值显存。

|   Param Key  | Type | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "gradient_checkpoint.type"    |  str    | ""   |  Gradient checkpoint选点方式，现提供两种方式， "collection": 用户指定GC tensor, "auto": epl 自动选点  | 
|     "gradient_checkpoint.end_taskgraph"    |  integer    |   -1  |    当开启auto GC，用于指定GC的结束taskgraph。   | 
|     "gradient_checkpoint.check_gradients"    |  bool    |   False |   校验GC生成的梯度的正确性。  | 

代码示例：

自动GC选点, 对于Transformer类模型，推荐使用auto GC的方式。

```python
import epl
# Enable auto GC.
config = epl.Config({"gradient_checkpoint.type": "auto"})
epl.init(config)
```

手动选点

```python
import tensorflow as tf
import epl
config = epl.Config({"gradient_checkpoint.type": "collection"})
epl.init(config)

# 手动指定checkpoint tensor
tensor = op1()
tf.add_to_collection("checkpoints", tensor)
```


##  Zero 配置

在数据并行的场景下，每个卡上会存放一个模型副本，optimizer state等，这些信息在每张卡上都是一样，存在很大的冗余量。当模型变大，很容易超出单卡的显存限制。

在分布式场景下，我们可以通过类似zero的思路，将optimizer state和gradient分片存在不同的卡上，从而减少单卡的persistent memory占用。

|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "zero.level"    |  str   |   ""  |  ZERO开启级别，目前EPL支持 level设置为 "v1",  对optimizer states和gradients进行拆分。|

```python
import epl

# 打开Zero
config = epl.Config({"zero.level": "v1"})
epl.init(config)
```

注意：
1. epl zero只能应用于数据并行部分。
2. 目前不支持zero组合gradient accumulation使用。
3. 支持的GPU cluster为多机一卡的场景，即多个worker，每个worker一张卡。

## Offload 配置

当模型参数量很大，超出GPU显存限制，我们可以通过CPU Offload，利用内存来扩大单卡可以训练的模型规模。
epl可以通过设置offload.level来实现offload。
- "v0": offload所有的参数到CPU上。

|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "offload.level"    |  str    |   ""  |     offload level.| 

示例
```python
import epl
config = epl.Config({"offload.level": "v0"})
epl.init(config)
```

## Memory-efficient AMP 配置

TF原生的AMP设计会在显存中保留一份FP16的weight，对于参数量很大的模型，会额外增加显存占用。为了让AMP显存开销更友好，EPL实现了一版memory-efficient AMP, 通过实时转换和释放的方式来节省显存。

用户可以通过配置amp.level 参数来开启EPL的AMP。

|   Param Key  | Type  | Default | Description |     
|:--------:|:------:|:-------:|:-----------:| 
|     "amp.level"    |  str    | ""  |    Auto mixed precision level, currently only support O1. | 
|     "amp.debug_log"    |  bool   |   False  |    Enable amp debug log.| 
|     "amp.loss_scale"    |  integer/str    |   "dynamic"  |   Loss scale for amp, can be "dynamic" or number(for fix).| 

示例

```python
import epl
config = epl.Config({"amp.level": "O1", "amp.loss_scale": "dynamic"})
# fixed loss scaling
config = epl.Config({"amp.level": "O1", "amp.loss_scale": 128})
epl.init(config)
```


## Optimizer 配置

训练任务在做参数更新的时候(optimizer apply), 对于一些有较多临时tensor buffer的optimizer实现，容易消耗较多的显存。可以通过配置num_apply_group参数实现分组apply的方式节省显存消耗。

|   Param Key  | Type  | Default | Description |    
|:--------:|:------:|:-------:|:-----------:| 
|     "optimizer.num_apply_group"    |  integer   |   1  | Number of gradient apply groups. |

示例

```python
import epl
config = epl.Config({"optimizer.num_apply_group": 30})
epl.init(config)
```

## Cluster 配置


|   Param Key  | Type  | Default | Description |    
|:--------:|:------:|:-------:|:-----------:| 
|     "cluster.device_place_prefer_intra_node"    |  bool    |   True  |  Prefer placing one model replica within node. | 
|     "cluster.run_visible_devices"    |  str   |   ""  |  Visible devices for session. Usually, its value is setted by scheduler. | 
|     "cluster.colocate_split_and_replicate"    |  bool   |   False  |  如果cluster.colocate_split_and_replicate设为True，不同的taskgraph会共享相同的device。 | 

## 通信配置


|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "communication.num_communicators"    |  integer   |   2  |    通信线程池的communicator个数。   | 
|     "communication.sparse_as_dense"    |  bool   |   False  |     是否将sparse tensor转换为dense tensor进行通信。   | 
|     "communication.max_splits"    |  integer    |   5  |    最大通信梯度融合的分组数。   | 
|     "communication.fp16"    |  bool     |   False  |    是否开启fp16参数通信。   | 
|     "communication.fp16_scale"    |  integer  |   128  |   开启fp16参数通信后，为防止梯度消失问题，梯度scale系数。   | 
|     "communication.clip_after_allreduce"    |  bool   |   False  |   选择通信后进行梯度Clip，还是在梯度Clip后进行通信。  | 
|     "communication.gradients_reduce_method"    |  str  |   "mean"  |   梯度AllReduce的方式，可以是 "mean" 和 "sum"。 |

## IO配置 


|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "io.slicing"    |  bool   |   False  | 是否自动对数据进行分片。   | 
|     "io.unbalanced_io_slicing"    |  bool   |   False  |   允许数据分片切分时worker分配的文件数目不相同，部分worker会多分配1个训练文件。   | 
|     "io.drop_last_files"    |  bool    |   False  |   对文件列表进行均匀切分，丢弃多余的文件。 |

## Auto Parallel配置


|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "auto.auto_parallel"    |  bool   |   False  |   是否打开自动并行化（目前还在实验阶段）。   | 
 
