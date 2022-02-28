#!/bin/bash
set -e

unset LD_PRELOAD

if [[ -z "${PYTHON}" ]]; then
    export PYTHON=$(which python)
fi
export PYTHONPATH=../python/:${PYTHONPATH}

export NET_DEVICE=$(find /sys/class/net/ -name 'bond0' -o -name 'eth*' | cut -d '/' -f5- | sort | head -1)

export RDMA_DEVICE=$(ls -1 /sys/class/infiniband/ 2>/dev/null | head -1)
export RDMA_SL=5
export RDMA_TRAFFIC_CLASS=136
export RDMA_QUEUE_DEPTH=4096

export NCCL_CHECKS_DISABLE=1
export NCCL_CHECK_POINTERS=1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INFO
export NCCL_LAUNCH_MODE=PARALLEL
export NCCL_MAX_NRINGS=2

export NCCL_IB_HCA=mlx5
export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3

export NCCL_SOCKET_IFNAME=${NET_DEVICE}
export NCCL_SOCKET_NTHREADS=2 # Same to AWS
export NCCL_NSOCKS_PERTHREAD=8 # Same to AWS
export NCCL_BUFFSIZE=2097152 # Avoid Cuda failure 'out of memory'

export GLOO_SOCKET_IFNAME=${NET_DEVICE}

export OMP_NUM_THREADS=3


if [ -n "$2" ]; then
  source $2
fi

${PYTHON} $1
