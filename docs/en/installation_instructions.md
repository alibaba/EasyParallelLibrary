# Installation

You can install EPL by following instructions.

## Requirements

- TensorFlow-GPU 1.15

## Install from source

### Build from NVIDIA TF1.15 DOCKER

```bash
nvidia-docker run -ti --gpus all --name build_epl_with_nvtf1.15_21.12 --net host --ipc host -v /mnt:/mnt nvcr.io/nvidia/tensorflow:21.12-tf1-py3 bash

# clone and install EPL
git clone https://github.com/alibaba/EasyParallelLibrary.git
cd EasyParallelLibrary
pip install .
```

### Build from TensorFlow TF1.15 DOCKER

```bash
nvidia-docker run -ti --gpus all --name build_epl_with_tf1.15 --net host --ipc host -v /mnt:/mnt tensorflow/tensorflow:1.15.5-gpu-py3 bash
# install nccl
apt update
apt install libnccl2 libnccl-dev

# clone and install EPL
git clone https://github.com/alibaba/EasyParallelLibrary.git
cd EasyParallelLibrary
pip install .
```