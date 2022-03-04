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

#include "communicators/nccl_communicator.h"

namespace tensorflow {
namespace communicators {

namespace {
const int64 kNumElements = NCCL_UNIQUE_ID_BYTES / sizeof(int64);
}  // anonymous namespace

REGISTER_OP("EplNcclCommunicatorGetId")
    .Output("id: int64")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->Vector(kNumElements));
      return Status::OK();
    })
    .SetIsStateful()
    .Doc(R"doc(
Get ID of the NCCL communciator.

id: Unique ID of the NCCL communicator.
)doc");

#if GOOGLE_CUDA
class NcclCommunicatorGetIdOp : public OpKernel {
 public:
  NcclCommunicatorGetIdOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    static_assert(NCCL_UNIQUE_ID_BYTES % sizeof(int64) == 0, "Unexpected");
    Tensor* id;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({kNumElements}), &id));
    ncclUniqueId nccl_id;
    ncclGetUniqueId(&nccl_id);
    std::memcpy(reinterpret_cast<char*>(id->flat<int64>().data()),
                nccl_id.internal, NCCL_UNIQUE_ID_BYTES);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("EplNcclCommunicatorGetId").Device(DEVICE_GPU).HostMemory("id"),
    NcclCommunicatorGetIdOp);

REGISTER_RESOURCE_HANDLE_OP(EplNcclCommunicator);

REGISTER_KERNEL_BUILDER(Name("EplNcclCommunicatorHandleOp").Device(DEVICE_GPU),
                        ResourceHandleOp<NcclCommunicator>);
#endif  // GOOGLE_CUDA

REGISTER_OP("EplNcclCommunicatorIsInitialized")
    .Output("is_initialized: bool")
    .Input("handle: resource")
    .SetShapeFn(tensorflow::shape_inference::ScalarShape)
    .Doc(R"doc(
Checks whether a NCCL communicator has been initialized.

is_initialized: True if the NCCL communicator is initialized.
handle: Handle of a NCCL communicator.
)doc");

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("EplNcclCommunicatorIsInitialized")
                            .Device(DEVICE_GPU)
                            .HostMemory("is_initialized")
                            .HostMemory("handle"),
                        IsResourceInitialized<NcclCommunicator>);
#endif  // GOOGLE_CUDA

REGISTER_OP("EplNcclCommunicatorCreater")
    .Input("handle: resource")
    .Input("id: int64")
    .Attr("shared_name: string")
    .Attr("size: int")
    .Attr("rank: int")
    .SetShapeFn(tensorflow::shape_inference::NoOutputs)
    .Doc(R"doc(
Creates a NCCL communicator and returns a handle to it.

handle: Handle of a NCCL communicator.
id: Unique ID of the NCCL communicator.
shared_name: Name of the communicator.
size: Total number of ranks in the communicator.
rank: Current rank in the communicator.
)doc");

#if GOOGLE_CUDA
class NcclCommunicatorCreaterOp : public OpKernel {
 public:
  explicit NcclCommunicatorCreaterOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &shared_name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("size", &size_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("rank", &rank_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* id;
    OP_REQUIRES_OK(ctx, ctx->input("id", &id));
    string nccl_id = string(id->tensor_data().data(), NCCL_UNIQUE_ID_BYTES);
    NcclCommunicator* comm = new NcclCommunicator();
    OP_REQUIRES_OK(ctx, comm->Create(nccl_id, shared_name_, size_, rank_));
    Status s = CreateResource(ctx, HandleFromInput(ctx, 0), comm);
    if (!s.ok() && s.code() != error::ALREADY_EXISTS) {
      OP_REQUIRES(ctx, false, s);
    }
  }

 private:
  string shared_name_;
  int size_;
  int rank_;
};

REGISTER_KERNEL_BUILDER(
    Name("EplNcclCommunicatorCreater").Device(DEVICE_GPU).HostMemory("id"),
    NcclCommunicatorCreaterOp);
#endif  // GOOGLE_CUDA

}  // namespace communicators
}  // namespace tensorflow
