# Parallelism Strategy API

In this section, we will introduce the parallelism primitive API,
which can be used to build various parallelism strategies.

Firstly, we will recap some basic concepts used in this document.
- *Model replica*: local DNN model (without parallelism or gradient accumulation).
- *micro batch size(mb)*: number of samples consumed by one model replica in each training iteration.
- *num_micro_batch*: number of micro batch used in pipeline or GA for each model replica in each training iteration.
- *global batch size*: Assume the model replica number is $N$, then the global batch size is `N * mb * num_micro_batch`.
- *TaskGraph*: TaskGraph is a subset of the model for parallel transformation and execution.

Unless otherwise specified, the default batch size of the local model is `micro batch size`.

## Parallel Strategy Primitive

With strategy primitive annotation, EPL partitions the model into multiple `TaskGraphs`
 and applies the parallelism strategies to the `TaskGraphs`.
EPL provides two basic strategy primitives: `replicate` and `split`.
Each strategy annotation generates one `TaskGraph`.

### replicate

`replicate` annotates operations to data parallelism, where each replica consumes different input data.
Operations defined under `replicate` scope form one `TaskGraph`.
1. If the whole model is annotated with `replicate`，i.e. there is one `TaskGraph`, then it is the same as the traditional data parallelism.
2. If part of the model is annotated with `replicate`, EPL will perform data parallelism for the corresponding TaskGraph.

API definition:

```python
replicate(device_count=None, name=None)
```

| Args | Required | Description |
|:----:|:---:|:-----------:|
| device_count | True | device count for one model replica defined under `replicate` scope. |
| name | | strategy name |

For data parallelism, one model replica is placed in one GPU (`device_count=1`), and EPL will infer the total number of replicas given the allocated number of GPUs.
When `device_count>1`, EPL will split the input batch into `device_count` parts when replicating the model, and keeps the total batch size of replicas the same as the original local batch size.

The following examples show data parallelism, where each model replica is placed in one GPU.
If the total allocated GPU number is 8, then the model will be scaled to 8 GPUs to perform data parallelism training.

```python
import epl
epl.init()
with epl.replicate(device_count=1):
  model()
```

### split

`split` annotates model to be split. Operations defined under `split` scope form a `TaskGraph`, which is split over multiple GPUs for parallel computation.

API definition:

```python
split(device_count=None, name=None)
```

| Args | Required | Description | 
|:----:|:---:|:-----------:|
| device_count | True | number of devices to split and place the model. |
| name |  | strategy name |

The following example shows the tensor model parallelism. The model is split over 8 GPUs.


```python
import epl
epl.init()
with epl.split(device_count=8):
  model()
```

## set_default_strategy
EPL also provides `set_default_strategy` to set the default parallelism strategies for operations.

```
set_default_strategy(strategy)
```
| Args | Required | Description | 
|:----:|:---:|:-----------:|
| strategy | True | parallelism strategy. |

The following example shows the data parallelism by setting the default strategy to `replicate`.
```python
import epl
epl.init()
epl.set_default_strategy(epl.replicate(device_count=1))
model()
```

## API Instruction
- By default, different TaskGraphs are placed in different devices.
- We do not allow nesting strategy annotations.
- Users only need to annotate the forward part of the model, the backward and apply operations are automatically co-located with the forward operations.

To learn how to use the above API to implement various parallelism strategies,
 such as pipeline parallelism or hybrid parallelism,
 please refer to [parallelism examples](api_examples.md).