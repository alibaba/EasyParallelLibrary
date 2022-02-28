# 流水并行

本节将介绍如何通过EPL来对Bert模型做 Pipeline 分布式训练。

## 训练准备

这个例子采用的Bert模型代码基于Google官方的Bert Repo https://github.com/google-research/bert 。

### 下载预训练Bert模型文件

```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

### 准备数据集

```
mkdir data
cd data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
```

## 分布式 Bert 流水并行训练

用户仅需要添加几行并行化策略和配置代码，即可实现Bert的流水并行训练策略。

```diff
+ import epl
+ epl.init(epl.Config({"pipeline.num_micro_batch": 4}))

# model annotation
+ epl.set_default_strategy(epl.replicate(1))
model_stage0()
+ epl.set_default_strategy(epl.replicate(1))
model_stage1()
```

用户可以通过以下脚本来启动一个2个stage的流水并行训练任务。

```
epl-launch --num_workers 1 --gpu_per_worker 2 scripts/train_bert_base_dp.sh
```

完整的训练代码可以参考 [EPL Bert Example](https://github.com/alibaba/FastNN/tree/master/bert)。


## 模型验证
完成训练后，你可以通过以下脚本来得到验证结果。

```bash
SQUAD_DIR=data
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ${output_dir}/predictions.json
```
在模型训练2 Epoch后，预期会得到 f1 ~= 88.0, exact_match ~= 79.8。

