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

REGISTER_OP("EplNcclCommunicatorAllReduce")
    .Output("output: T")
    .Input("handle: resource")
    .Input("input: T")
    .Attr("size: int >= 1 = 1")
    .Attr("rank: int >= 0 = 0")
    .Attr("reduce_op: int >= 0 = 0")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .SetIsStateful()
    .Doc(R"doc(
Allreduce using a NCCL communicator.

output: A reduced tensor.
handle: Handle of a NCCL communicator.
input: A tensor to reduce.
size: Total number of devices in the communicator.
rank: Index of current device in the communicator.
reduce_op: Reduce ops: 0 for SUM, 1 for PROD, 2 for MAX, 3 for MIN.
)doc");

#if GOOGLE_CUDA
template <typename T>
class NcclCommunicatorAllReduceOp : public NcclCommunicatorAsyncOp {
 public:
  explicit NcclCommunicatorAllReduceOp(OpKernelConstruction* ctx)
      : NcclCommunicatorAsyncOp(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("reduce_op", &reduce_op_));
    OP_REQUIRES(ctx, reduce_op_ >= 0 && reduce_op_ < 4,
                errors::InvalidArgument("reduce_op should be 0 for SUM, 1 for "
                                        "PROD, 2 for MAX or 3 for MIN"));
  }

  void ComputeAsyncWithCommunicator(NcclCommunicator* comm,
                                    OpKernelContext* ctx,
                                    DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);

    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input->shape(), &output),
                         done);
    se::Event* outputs_allocated = RecordEventOnTensorStream(ctx);
    WaitThenDeleteEventOnStream(outputs_allocated);

    VLOG(1) << comm->DebugString() << " [" << name() << "] [AllReduce]";
    OP_REQUIRES_OK_ASYNC(
        ctx, comm->AllReduce<T>(output, input, reduce_op_, cuda_stream()),
        done);
    done();
  }

 private:
  int reduce_op_;
};

#define REGISTER_KERNEL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("EplNcclCommunicatorAllReduce") \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<TYPE>("T"),      \
                          NcclCommunicatorAllReduceOp<TYPE>);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

}  // namespace communicators
}  // namespace tensorflow
