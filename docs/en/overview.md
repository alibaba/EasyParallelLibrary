# Easy Parallel Library

## Overview

Easy Parallel Library (EPL) is a general and efficient deep learning framework for distributed giant model training.
- Usability - Users can implement different parallelism strategies with a few lines of annotations, including data parallelism, pipeline parallelism, tensor model parallelism, and their hybrids. 
- Memory Efficient - EPL provides various memory-saving techniques, including gradient checkpoint, ZERO, CPU Offload, etc. Users are able to train larger models with fewer computing resources.
- High Performance - EPL provides an optimized communication library to achieve high scalability and efficiency.


## Examples

Here are a few examples of different parallelism strategies by changing only annotations.
Please refer to the full [tutorials](tutorials/index.rst) for more examples.

### Data Parallelism

The following example shows a basic data parallelism annotation.
The data parallelism degree is determined by the allocated GPU number.

```diff
+ import epl
+ epl.init()
+ with epl.replicate(device_count=1):
    model()
```


### Pipeline Parallelism

The following example shows pipeline parallelism with two pipeline stages, each stage is computed with one GPU.
If the total GPU number is 4, EPL will automatically apply two-degree data parallelism over the model pipeline.

```diff
+ import epl
+ 
+ config = epl.Config({"pipeline.num_micro_batch": 4})
+ epl.init(config)
+ with epl.replicate(device_count=1, name="stage_0"):
    model_part1()
+ with epl.replicate(device_count=1, name="stage_1"):
    model_part2()
```

### Tensor Model Parallelism
The following example shows a tensor model parallelism annotation.
We apply data parallelism to the `ResNet` part, and apply tensor model parallelism to `classification` part.

```diff
+ import epl
+ config = epl.Config({"cluster.colocate_split_and_replicate": True})
+ epl.init(config)
+ with epl.replicate(8):
    ResNet()
+ with epl.split(8):
    classification()
```


## Publication

If you use EPL in your publication, please cite it by using the following BibTeX entry.

```BibTeX
@misc{jia2021whale,
      title={Whale: Scaling Deep Learning Model Training to the Trillions}, 
      author={Xianyan Jia and Le Jiang and Ang Wang and Jie Zhang and Xinyuan Li and Wencong Xiao and Langshi chen and Yong Li and Zhen Zheng and Xiaoyong Liu and Wei Lin},
      year={2021},
      eprint={2011.09208},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```