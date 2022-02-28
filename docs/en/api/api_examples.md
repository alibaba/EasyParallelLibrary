## Parallelism API Examples

In this section, we will introduce how to use EPL [parallelism strategy APIs]((strategy.md))
to implement different parallelism strategies, as well as their hybrids.

### Data Parallelism

The following snippet shows the data parallelism, where each model replica is placed in one GPU.
If the user uses 8 GPUs, then it is a data parallelism task with 8 replicas.

```python
import epl
epl.init()
with epl.replicate(device_count=1):
  model()
```

### Pipeline Parallelism

In the following example, the model is divided into two `TaskGraph`s, i.e., "stage_0" and "stage_1".
We can set the number of micro batches of the Pipeline by configuring the `pipeline.num_micro_batch` parameter.
This model requires two GPUs to place "stage_0" and "stage_1" for each model replica.
If the task uses 8 GPUs, EPL will automatically apply a 4-degree data parallelism over the pipeline.

```python
import epl

config = epl.Config({"pipeline.num_micro_batch": 4})
epl.init(config)
with epl.replicate(device_count=1, name="stage_0"):
  model_part1()
with epl.replicate(device_count=1, name="stage_1"):
  model_part2()
```

### Tensor Model Parallelism

#### Large-scale Image Classification

The following example applies different strategies to different parts of the model.
We apply data parallelism for the `resnet` part and apply tensor model parallelism to the `classification` part.
To reduce the communication overhead among the two taskgraphs, we set `cluster.colocate_split_and_replicate` to 
colocate the two taskgraphs to the same devices.

```python
import epl
config = epl.Config({"cluster.colocate_split_and_replicate": True})
epl.init(config)
with epl.replicate(8):
  resnet()
with epl.split(8):
  classification()
```

#### MOE Transformer

The following example shows the implementation of a MoE model.
We split the tensors for MoE, and set the default strategy as `replicate` for the remaining operations.

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
