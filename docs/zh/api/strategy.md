# 并行化接口

本文档主要介绍了EPL并行化原语接口定义，和接口使用的注意事项。

在开始介绍接口定义之前，用户需要了解以下基本概念：
- *模型副本* ：用户定义的单机单卡模型（不包含任何并行化和GA操作）。
- *micro batch size(mb)*: 一个*模型副本*训练迭代一步学习的samples数量。
- *num_micro_batch*: 一个模型副本 GA或pipeline 累计的micro batch数量。
- *global batch size*: 假设我们对一个模型副本做数据并行操作，并行度为N，则global batch size为 `N * mb * num_micro_batch`.
- *TaskGraph*: TaskGraph是一个并行化子图。

如果没有特殊说明，EPL默认用户定义的batch size为 `micro batch size`。

## Parallel Strategy 原语

EPL通过strategy annotation的方式来划分模型为多个`TaskGraph`，并在此基础上进行并行化。
EPL有两类strategy：`replicate` 和 `split`。每个strategy会定义一个`TaskGraph`。

### replicate

`replicate` 可以实现模型的数据并行计算，即将模型复制多份，每份模型副本消费不同数据，来实现计算的并行化。 `replicate` scope下的子模型会构成一个`TaskGraph`。
1. 当整个模型都标记了`replicate`，当前只有一个TaskGraph做复制，就是传统的数据并行模式。
2. 当部分模型标记了`replicate`, EPL会对这部分TaskGraph做复制。

接口定义：
```python
replicate(device_count=None, name=None)
```

| Args | Required | Description |
|:----:|:---:|:-----------:|
| device_count | True | `replicate` scope下一个模型副本用来计算的卡数。 |
| name | | strategy name |

对于数据并行，一个模型副本用一张卡来计算，EPL会根据当前资源总数推断出全局的副本数。
当`device_count`大于1的时候，EPL在做模型复制的时候会对micro batch size进行拆分，平均到`device_count`张卡上，保持用户模型的micro batch size保持不变。

示例：
```python
import epl
epl.init()
with epl.replicate(device_count=1):
  model()
```
上面这个例子是一个数据并行的例子，每个模型副本用一张卡来计算。如果用户申请了8张卡，就是一个并行度为8的数据并行任务。

### split

`split` 可以实现模型的tensor拆分计算。`split` scope下定义的子模型会构成一个`TaskGraph`，该`TaskGraph`会被拆分后放置在多张卡上计算。

接口定义：
```python
split(device_count=None, name=None)
```
| Args | Required | Description | 
|:----:|:---:|:-----------:|
| device_count | True | split 对应的taskgraph拆分到device_count张卡上计算 |
| name |  | strategy name |

示例：
```python
import epl
epl.init()
with epl.split(device_count=8):
  model()
```
上述例子将模型拆分到8张卡上做计算，如果用户申请了16张卡，EPL会默认在拆分模型外面嵌套一层数据并行。

## set_default_strategy
除了两个基本的并行化接口`replicate`和`split`外，EPL也提供了一个设置默认strategy的辅助接口，
如果用户调用set_default_strategy方法，会设置一个默认的并行化策略和对应的`TaskGraph`。
这个接口可以帮助模型并行化表达更简洁，同时更灵活地表达出复杂的并行策略。

接口定义：
```
set_default_strategy(strategy)
```
| Args | Required | Description | 
|:----:|:---:|:-----------:|
| strategy | True | 并行化策略。 |

示例：
```python
import epl
epl.init()
epl.set_default_strategy(epl.replicate(device_count=1))
model()
```
上述例子设置了一个默认的`replicate`策略，通过这种方式也可以实现模型的数据并行。

##  接口使用说明与要求
- 不同并行化strategy生成的TaskGraph默认会放置在不同Device上。
- Strategy annotation不允许嵌套。
- 用户只需标记模型前向代码，backward和apply自动与Forward colocate在对应的`TaskGraph`里。

关于如何使用并行化接口实现更多灵活的并行化策略，比如Pipeline，混合并行等，您可以继续阅读 [并行化例子](api_examples.md)。
