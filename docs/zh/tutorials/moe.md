# MoE算子拆分并行

本节将介绍如何通过EPL来实现 MoE (Mixture of Experts) transformer 模型训练。

## 训练准备

模型代码将基于[tensor2tensor](https://github.com/tensorflow/tensor2tensor)的组件。

### 准备数据集

```
t2t-datagen --data_dir=data --tmp_dir=data/original/dataset --problem=translate_ende_wmt32k
```
或者，通过在`scripts/train_moe_t5.sh`脚本中设置`FLAGS.generate_data`来自动下载和准备数据。

详细的准备流程可以参考 [tensor2tensor文档](https://github.com/tensorflow/tensor2tensor#adding-a-dataset).

## 分布式训练
EPL仅需添加几行代码来实现 MoE 算子拆分并行，如下所示：

```diff
+ import epl
+ config = epl.Config({"cluster.colocate_split_and_replicate": True})
+ epl.init(config)
+ epl.set_default_strategy(epl.replicate(total_gpu_num))

AttentionAndGating()

+ with epl.split(total_gpu_num):
  MOE_Variable_Define()

MOE_Calculation_Define()
```

用户可以通过以下脚本来启动一个2卡的MOE算子拆分并行训练任务。

```
epl-launch --num_workers 2 --gpu_per_worker 1 scripts/train_moe_t5.sh
```

完整的训练代码可以参考 [EPL MOE Example](https://github.com/alibaba/FastNN/tree/master/moe/)。