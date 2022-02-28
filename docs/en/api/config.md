# Configuration

Users can enable EPL optimized features by configuration.

The configuration tables include:
- Param Key: parameter name, which is defined in the format of "param_category.attribute". `param_category` is the category of parameters，e.g., `pipeline`. The `attribute` is the detailed configuration attribute, e.g., `num_micro_batch`.
- Type: parameter type, e.g. str/float/integer/bool
- Default: default value
- Description: parameter description

Configuration APi:

```python
Config(param_dict=None)
```

| Args | Type| Required | Description |
|:----:|:----:|:---:|:-----------:|
| param_dict | dict | False | Parameter dict, where key is the parameter key and value is the parameter value. |

Example:

```python
import epl
config = epl.Config({"pipeline.num_micro_batch": 4})
epl.init(config)
```
In the above example, we set the configuration by passing a `param_dict`.

You can refer to the following configuration tables for the full parameters.


## Pipeline Configuration
|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "pipeline.num_micro_batch"    |  integer    |   1  |   Pipeline number of micro batches.  | 
|     "pipeline.num_stages"    |  integer    |     None  |     If `auto.auto_parallel` is True, you can set `pipeline.num_stages` to automatically partition pipeline stages. | 
|     "pipeline.strategy"   |  str    |   "PreferBackward"  |     Pipeline schedule policies can be one of ["PreferBackward", "PreferForward"] | 

- `PreferBackward`: pipeline strategy similar to [PipeDream](https://www.microsoft.com/en-us/research/publication/pipedream-generalized-pipeline-parallelism-for-dnn-training/).
- `PreferForward`: pipeline strategy similar to [GPipe](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html).

## Gradient Checkpoint (GC) Configuration

Gradient checkpoint reduces the peak memory by saving the activation memory consumption through re-computation.

|   Param Key  | Type | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "gradient_checkpoint.type"    |  str    | ""   |  Type to select checkpoint tensor, can be one of ("collection", "auto"). <br/>"collection": user selected GC tensors. <br/>"auto": automatically searching the GC tensors by analyzing the model.  | 
|     "gradient_checkpoint.end_taskgraph"    |  integer    |   -1  |    The last taskgraph index to apply GC.   | 
|     "gradient_checkpoint.check_gradients"    |  bool    |   False |   Validate the GC gradients.  | 

Examples:

Automatic GC works well for Transformer models.

```python
import epl
# Enable auto GC.
config = epl.Config({"gradient_checkpoint.type": "auto"})
epl.init(config)
```

Users can also specify the checkpoint tensors by adding them to a collection, shown as follows:

```python
import tensorflow as tf
import epl
config = epl.Config({"gradient_checkpoint.type": "collection"})
epl.init(config)

# specify a checkpoint tensor
tensor = op1()
tf.add_to_collection("checkpoints", tensor)
```


##  Zero Configuration

ZeRO leverages the aggregate computation and memory resources of data parallelism to reduce the memory and compute requirements of each device (GPU) used for model training.
You can refer to [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) for more information. 

|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "zero.level"    |  str   |   ""  |  ZeRO level，EPL now supports "v1",  which partitions the optimizer states and gradients. |

```python
import epl

config = epl.Config({"zero.level": "v1"})
epl.init(config)
```

Note：
1. EPL ZeRO works only for data parallelism.
2. Now ZeRO cannot be used with gradient accumulation.
3. ZeRO only works for GPU cluster of Nx1 configuration, i.e., N workers, and each worker with one GPU.

## Offload Configuration

EPL supports training large models by offloading weight to CPU memory.

Users can offload parameters by setting `offload.level`.
- "v0": offload all weight to CPU.

|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "offload.level"    |  str    |   ""  |     offload level.| 

Example:

```python
import epl
config = epl.Config({"offload.level": "v0"})
epl.init(config)
```

## Memory-efficient AMP Configuration


Memory-efficient AMP does not keep the FP16 weight in memory, instead, EPL casts the weight when needed.

Users can enable EPL AMP by setting `amp.level`.

|   Param Key  | Type  | Default | Description |     
|:--------:|:------:|:-------:|:-----------:| 
|     "amp.level"    |  str    | ""  |    Auto mixed precision level, currently only supports O1. | 
|     "amp.debug_log"    |  bool   |   False  |    Enable amp debug log.| 
|     "amp.loss_scale"    |  integer/str    |   "dynamic"  |   Loss scale for amp, can be "dynamic" or number(for fix).| 

Example:

```python
import epl
config = epl.Config({"amp.level": "O1", "amp.loss_scale": "dynamic"})
# fixed loss scaling
config = epl.Config({"amp.level": "O1", "amp.loss_scale": 128})
epl.init(config)
```


## Optimizer Configuration

Optimizer-related configuration.
When updating the parameters in the training process, some user-defined optimizers will consume a large amount of temporary tensor buffers,
which increases the peak memory a lot. We can set `num_apply_group` to save memory by updating parameters in groups.

|   Param Key  | Type  | Default | Description |    
|:--------:|:------:|:-------:|:-----------:| 
|     "optimizer.num_apply_group"    |  integer   |   1  | Number of gradient apply groups. |

Example:

```python
import epl
config = epl.Config({"optimizer.num_apply_group": 30})
epl.init(config)
```

## Cluster Configuration


|   Param Key  | Type  | Default | Description |    
|:--------:|:------:|:-------:|:-----------:| 
|     "cluster.device_place_prefer_intra_node"    |  bool    |   True  |  Prefer placing one model replica within node. | 
|     "cluster.run_visible_devices"    |  str   |   ""  |  Visible devices for session. Usually, its value is set by the scheduler. | 
|     "cluster.colocate_split_and_replicate"    |  bool   |   False  |  If cluster.colocate_split_and_replicate is set to True，different taskgraphs will be co-locate in the same device. | 

## Communication Configuration


|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "communication.num_communicators"    |  integer   |   2  |    number of communicators.  | 
|     "communication.sparse_as_dense"    |  bool   |   False  |     Whether to transform sparse tensor to dense tensor before communication.   | 
|     "communication.max_splits"    |  integer    |   5  |   Max number of communication groups for tensor fusion. | 
|     "communication.fp16"    |  bool     |   False  |    Enable FP16 AllReduce.  | 
|     "communication.fp16_scale"    |  integer  |   128  |   Scale the gradients after FP16 AllReduce.   | 
|     "communication.clip_after_allreduce"    |  bool   |   False  |   Clip gradients after AllReduce.  | 
|     "communication.gradients_reduce_method"    |  str  |   "mean"  |   AllReduce type, can be one of ("mean", "sum") |

## IO Configuration


|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "io.slicing"    |  bool   |   False  | Whether to slice the dataset.   | 
|     "io.unbalanced_io_slicing"    |  bool   |   False  |   Allow unbalanced dataset slicing.   | 
|     "io.drop_last_files"    |  bool    |   False  |  Partition the data files evenly, and drop the last files that cannot be divided. |

## Auto Parallel Configuration


|   Param Key  | Type  | Default | Description |   
|:--------:|:------:|:-------:|:-----------:| 
|     "auto.auto_parallel"    |  bool   |   False  |   Whether to enable automatic parallelism. (Experimental)   | 
 
