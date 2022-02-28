# Pipeline Parallelism

In this section, we will show how to scale the training of Bert model with EPL pipeline parallelism.


## Training setup.

The model code is based on https://github.com/google-research/bert .

### Get pretrained bert base model.

```
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
unzip uncased_L-12_H-768_A-12.zip
```

### Prepare dataset

```
mkdir data
cd data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
wget https://raw.githubusercontent.com/allenai/bi-att-flow/master/squad/evaluate-v1.1.py
```

## Distributed Bert training

### Pipeline parallelism

To implement Bert pipeline parallelism, EPL only needs to change the annotation and configuration, as follows:

```diff
+ import epl
+ epl.init(epl.Config({"pipeline.num_micro_batch": 4}))

# model annotation
+ epl.set_default_strategy(epl.replicate(1))
model_stage0()
+ epl.set_default_strategy(epl.replicate(1))
model_stage1()
```

You can refer to [EPL Bert Example](https://github.com/alibaba/FastNN/tree/master/bert) for detailed implementation.

The following command launches a pipeline parallelism program with two stages.

```
epl-launch --num_workers 1 --gpu_per_worker 2 scripts/train_bert_base_dp.sh
```

## Evaluation
After training, you can perform the following commands to get the evaluation results.

```bash
SQUAD_DIR=data
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json ${output_dir}/predictions.json
```
You are expected to get f1 ~= 88.0, exact_match ~= 79.8 after 2 epochs.

