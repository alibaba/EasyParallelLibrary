# MoE Tensor Model Parallelism

This repo contains MoE (Mixture of Experts) transformer training examples with EPL.

## Training setup.
The model code is based on https://github.com/tensorflow/tensor2tensor .

### Prepare dataset

Refering to https://github.com/tensorflow/tensor2tensor#adding-a-dataset, script for `translate_ende_wmt32k` shows as following:

```
t2t-datagen --data_dir=data --tmp_dir=data/original/dataset --problem=translate_ende_wmt32k
```

Or, set `FLAGS.generate_data` in `scripts/train_moe_t5.sh` to generate dataset for problem `FLAGS.problem` automatially.

## Distributed Training

To implement MoE tensor model parallelism,
EPL only needs to change the annotation and configuration, as follows:


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

You can refer to [EPL MOE Example](https://github.com/alibaba/FastNN/tree/master/moe/) for detailed implementation.

The following command launches a tensor model parallelism program with two workers.

```
epl-launch --num_workers 2 --gpu_per_worker 1 scripts/train_moe_t5.sh
```


