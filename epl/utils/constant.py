# Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


DATASET_API_OPS = ["FIFOQueueV2", "TensorSliceDataset", "TableRecordReader", \
                   "TableRecordReaderV2", "TableRecordDataset"]
DATASET_OPS = ["IteratorGetNext", "QueueDequeueManyV2", "QueueDequeueMany"]
ODPS_TABLE_API_OPS = ["TableRecordReader", "TableRecordReaderV2", "TableRecordDataset"]
PAI_DATA_FORMAT = ["odps", "oss"]
PAI_DATA_PREFETCH = ["DataBufferTake"]
EXCLUDED_DEPEND_OPS = ["IsVariableInitialized", "MergeSummary"]
HOST_DEVICE = "CPU"
DEFAULT_DEVICE = "GPU"
DEFAUT_PIPELINE_STRATEGY = "PreferBackward"
# Workround. Delete when graph execution order is ok.
OPS_IGNORED_FOR_ENT_SCHEDULER = \
    ["Const", "Shape", "BroadcastGradientArgs", \
     "DynamicStitch", "Pack", "L2Loss", "UnsortedSegmentSum", \
     "Reshape", "ExpandDims", "StridedSlice", "Transpose", \
     "Abs", "Acos", "Acosh", "ArgMax", "ArgMin", "Asin", "Asinh", \
     "Tan", "Atan", "Atanh", "BesselI0e", "BesselI1e", "Bincount", \
     "Ceil", "Cos", "Cosh", "Cumprod", "Cumsum", "Digamma", \
     "Erf", "Erfc", "Exp", "Expm1", "Floor", "InvertPermutation", \
     "IsFinite", "IsInf", "IsNan", "All", "Lgamma", "Log", "Log1p", \
     "Neg", "Real", "RsqrtGrad", "Polygamma", "Reciprocal", "Rint", \
     "Round", "Rsqrt", "Sigmoid", "Sign", "Sin", "Sinh", "Softplus", \
     "Sqrt", "Square", "Tan", "Tanh", "UnsortedSegmentMax", \
     "UnsortedSegmentMin", "RealDiv", "UnsortedSegmentProd", "AddN"]

OPS_IGNORED_FOR_EXIT_SCHEDULER = \
    ["Const", "Shape", "BroadcastGradientArgs", \
     "DynamicStitch", "Pack", "L2Loss", "StridedSlice"]

DEFAULT_TASK_NAME = "worker"
CHIEF_WORKER_NAME = "chief"
DEFAULT_GROUP_LEADER = "/job:worker/replica:0/task:0"
INPUT_FILE_TYPE = ["DT_STRING", 7]
REPLICA_PREFIX_FORMAT = "EPL_REPLICA_{}/"
MICRO_BATCH_PREFIX_FORMAT = "EPL_MICRO_BATCH_{}/"
MERGED_REPLICAS_SUFFIX = "_replicated"
PARALLEL_STRATEGY = "EPL_PARALLEL_STRATEGY"

# Function type
FUNCTION_TYPE = ["f", "dataset_factory"]

# Summary type
SUMMARY_AUDIO_TYPE = "SUMMARY_AUDIO_TYPE"
SUMMARY_HISTOGRAM_TYPE = "SUMMARY_HISTOGRAM_TYPE"
SUMMARY_IMAGE_TYPE = "SUMMARY_IMAGE_TYPE"
SUMMARY_SCALAR_TYPE = "SUMMARY_SCALAR_TYPE"
SUMMARY_TEXT_TYPE = "SUMMARY_TEXT_TYPE"
SUMMARY_TENSOR_TYPE = "SUMMARY_TENSOR_TYPE"
SUMMARY_TYPE = ["ScalarSummary", "AudioSummary", "HistorySummary", \
                "ImageSummary", "TextSummary", "TensorSummary", "MergeSummary"]

# Key for saving parallel information.
INFO_KEY_START_DIM = "INFO_KEY_START_DIM"
INFO_KEY_PREVENT_RUN_HOOK = "INFO_KEY_PREVENT_RUN_HOOK"

REDUCE_METHOD_MEAN = "mean"
REDUCE_METHOD_SUM = "sum"

# Default config for communication
DEFAULT_COM_MAX_SPLITS = 60
DEFAULT_COM_SPLIT_SIZE = 33554432 # 32MB

# PAI metrics
DISTRIBUTED_FRAMEWORK = "distributed_framework"
DISTRIBUTED_FRAMEWORK_NAME = "epl"

# Parallel information
PARALLEL_NUM_STAGES = "NUM_STAGES"
ALL_COMM_RESOURCES = "ALL_COMM_RESOURCES"

# Gradient checkpoint
GC_COLLECTION = "collection"
GC_AUTO = "auto"
GC_COLLECTION_NAME = "checkpoints"
GC_DST_SCOPE_NAME = "EPL_GRADIENT_CHECKPOINTS"
GC_AVOID_RECOMPUTE_OPS = ["EplNcclCommunicatorAllToAll", "EplNcclCommunicatorAllToAllv"]

# Offload
OFFLOAD_SCOPE_NAME = "EPL_OFFLOAD"

# Key for einsum distribution.
# TODO(jiangle.jl): Remove when auto split is ready.
INFO_EINSUM_INDEX = "INFO_EINSUM_INDEX"
SHARED_COMMUNICATOR_FOR_DISPATCH_AND_COMBINE = "DISPATCH_AND_COMBINE"
NUM_EINSUM_IN_SPLIT_FOR_MOE = 3


# Dataset-related files
dataset_related_files = ["data/ops/readers.py", "data/ops/dataset_ops.py"]

# Auto Mixed Precision
LOSS_SCALE_SCOPE_NAME = "UNSCALE_GRADIENTS"

ENCODING = "utf-8"

ENV_TF_CONFIG = "TF_CONFIG"

EPL_AMP_SUFFIX = "_EPL_AMP"
EPL_AMP_LOSS_SCALE_SUFFIX = "_EPL_LOSS_SCALE"

# Pipeline related
# mininum repeated model blocks for auto pipeline.
MIN_REPEAT_BLOCKS = 4
# auto stage policy const
AUTO_STAGE_POLICY_BALANCE_OP_NUM = "AUTO_STAGE_POLICY_BALANCE_OP_NUM"
AUTO_STAGE_POLICY_REPEATED_LAYERS = "AUTO_STAGE_POLICY_REPEATED_LAYERS"
AUTO_STAGE_POLICY_HEURISTIC = "AUTO_STAGE_POLICY_HEURISTIC"
