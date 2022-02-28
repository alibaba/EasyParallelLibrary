/* Copyright 2021 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma GCC system_header

#ifndef TENSORFLOW_H_
#define TENSORFLOW_H_

#define EIGEN_USE_THREADS
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/util.h"
#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1014L
#include "tensorflow/core/framework/bounds_check.h"
#else
#include "tensorflow/core/kernels/bounds_check.h"
#endif

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if GOOGLE_CUDA
#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/platform/cuda.h"
#include "tensorflow/stream_executor/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1008L
namespace se = ::perftools::gputools;
#endif

#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) >= 1014L
#define TF_RESOURCE_DEBUG_STRING_CONST const
#else
#define TF_RESOURCE_DEBUG_STRING_CONST
#endif

}  // namespace tensorflow

#endif  // TENSORFLOW_H_
