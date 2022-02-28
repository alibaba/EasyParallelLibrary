# Data Parallelism

In this section, we will show how to scale the training of ResNet-50 model with EPL data parallelism.


EPL can easily transform the local bert training program to a distributed one by adding a few lines of code.

```diff
+ import epl
+ epl.init()
+ epl.set_default_strategy(epl.replicate(device_count=1))

ResNet50()
training_session()
```

The following command launches a data parallelism program with two model replicas over two GPUs.
```
epl-launch --num_workers 2 --gpu_per_worker 1 scripts/train_dp.sh
```
`scripts/train_bert_base_dp.sh` is a local training script,
`epl-launch` will automatically launch a distributed training program by configuring cluster information.

You can refer to [EPL ResNet Example](https://github.com/alibaba/FastNN/tree/master/resnet/) for detailed implementation.



