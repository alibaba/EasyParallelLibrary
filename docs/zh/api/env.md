# Env

本文档主要介绍EPL的Env中获取常用的运行时信息。

你可以通过`epl.Env.get()` 获取当前的env对象。

## cluster

cluster包含当前分布式任务的集群信息。

|   Attribute  | Type  | Description |    
|:--------:|:-------:|:-----------:| 
|     `cluster.worker_num`    |  int    | worker数量 | 
|     `cluster.worker_index`    |  int   |   当前worker的index |

示例

```python
import epl
env = epl.Env.get()
worker_num = env.cluster.worker_num
worker_index = env.cluster.worker_index
```

## config

config包含当前epl的配置信息。

示例

```
import epl
env = epl.Env.get()
config = env.config
```


