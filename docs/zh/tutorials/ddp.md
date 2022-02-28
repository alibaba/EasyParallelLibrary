# 数据并行

本节将介绍如何通过EPL来对`ResNet-50`模型做数据并行分布式训练。

通过添加以下几行代码，EPL即可将本地训练程序转换成分布式训练程序。

```diff
+ import epl
+ epl.init()
+ epl.set_default_strategy(epl.replicate(device_count=1))

ResNet50()
training_session()
```


用户可以通过以下脚本来启动一个2卡的数据并行训练任务。

```
epl-launch --num_workers 2 --gpu_per_worker 1 scripts/train_dp.sh
```
`scripts/train_dp.sh` 是一个本地的训练脚本。

完整的训练代码可以参考[EPL ResNet Example](https://github.com/alibaba/FastNN/tree/master/resnet/)。


