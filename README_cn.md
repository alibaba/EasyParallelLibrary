[![pypi](https://img.shields.io/pypi/v/pyepl.svg)](https://pypi.org/project/pyepl)
[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://easyparallellibrary.readthedocs.io/zh/latest/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/EasyParallelLibrary/blob/main/LICENSE)

简体中文 | [English](README.md)

# 概览

Easy Parallel Library (EPL) 是一个高效易用的分布式模型训练框架。
EPL提供了简单易用的API来表达各种并行化策略，
用户仅需几行代码就可以轻松支持各种模型的高性能分布式训练。

EPL深度集成了各种训练优化技术，帮助更多的用户低成本，高性能，轻松开启大模型训练。
- 支持各种并行化策略及混合并行，用户仅通过转换并行化接口来实现不同并行化策略训练。
- 支持各种显存优化技术，包含自动Gradient Checkpoint，ZERO，CPU Offload技术等，帮助用户用更少的资源训练更大的模型。
- 支持通信优化技术，实现高效的分布式扩展性。

EPL助力了最大的中文多模态模型M6实现大规模分布式训练，通过512卡即可训练10万亿参数模型。

您可以阅读[**文档**](https://easyparallellibrary.readthedocs.io/zh/latest/)了解更多信息。同时EPL [**Model Zoo**](https://github.com/alibaba/FastNN) 提供了完整的分布式训练案例。


## 使用EPL添加分布式策略

通过添加几行代码，用户即可实现不同的并行化策略。完整的API介绍和并行化例子详见[API](https://easyparallellibrary.readthedocs.io/zh/latest/api/index.html)。
你也可以参考[使用教程](https://easyparallellibrary.readthedocs.io/zh/latest/tutorials/index.html) 来训练EPL模型库例子。

数据并行
```diff
+ import epl
+ epl.init()
+ with epl.replicate(device_count=1):
    model()
```


流水并行
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
在上述例子中，模型被切分成2部分，用户可以通过配置`pipeline.num_micro_batch`参数来设定Pipeline的micro batch数量。

算子拆分
```diff
+ import epl
+ config = epl.Config({"cluster.colocate_split_and_replicate": True})
+ epl.init(config)
+ with epl.replicate(8):
    resnet()
+ with epl.split(8):
    classification()
```
在上述例子中，我们对ResNet模型部分进行数据并行，对分类层进行算子拆分。


## Citation

```latex
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

## 联系我们

欢迎给我们提issue, 或者加入EPL官方钉钉群。

![DingTalk Group](docs/images/ding-group.png)
