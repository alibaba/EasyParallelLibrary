[![pypi](https://img.shields.io/pypi/v/pyepl.svg)](https://pypi.org/project/pyepl)
[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://easyparallellibrary.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/EasyParallelLibrary/blob/main/LICENSE)

English | [简体中文](README_cn.md)

# Easy Parallel Library

## Overview

Easy Parallel Library (EPL) is a general and efficient library for distributed model training.
- Usability - Users can implement different parallelism strategies with a few lines of annotations, including data parallelism, pipeline parallelism, tensor model parallelism, and their hybrids. 
- Memory Efficient - EPL provides various memory-saving techniques, including gradient checkpoint, ZERO, CPU Offload, etc. Users are able to train larger models with fewer computing resources.
- High Performance - EPL provides an optimized communication library to achieve high scalability and efficiency.

For more information, you may [read the docs](https://easyparallellibrary.readthedocs.io/en/latest/).

EPL [Model Zoo](https://github.com/alibaba/FastNN) provides end-to-end parallel training examples.

## Installation

To install EPL, please refer to the following [instructions](https://easyparallellibrary.readthedocs.io/en/latest/installation_instructions.html).

## Examples

Here are a few examples of different parallelism strategies by changing only annotations.
Please refer to [API documentation](https://easyparallellibrary.readthedocs.io/en/latest/api/index.html) for API details and [tutorials](https://easyparallellibrary.readthedocs.io/en/latest/tutorials/index.html) for more examples.

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
@inproceedings {jia2022whale,
	author = {Xianyan Jia and Le Jiang and Ang Wang and Wencong Xiao and Ziji Shi and Jie Zhang and Xinyuan Li and Langshi Chen and Yong Li and Zhen Zheng and Xiaoyong Liu and Wei Lin},
	title = {Whale: Efficient Giant Model Training over Heterogeneous {GPUs}},
	booktitle = {2022 USENIX Annual Technical Conference (USENIX ATC 22)},
	year = {2022},
	isbn = {978-1-939133-29-57},
	address = {Carlsbad, CA},
	pages = {673--688},
	url = {https://www.usenix.org/conference/atc22/presentation/jia-xianyan},
	publisher = {USENIX Association},
	month = jul,
}
```

## Contact Us

Join the Official Discussion Group on DingTalk.

![DingTalk Group](docs/images/ding-group.png)
