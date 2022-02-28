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

REGISTER_OP("EplNcclCommunicatorReduceScatter")
    .Output("output: T")
    .Input("handle: resource")
    .Input("input: T")
    .Attr("size: int >= 1 = 1")
    .Attr("rank: int >= 0 = 0")
    .Attr("reduce_op: int >= 0 = 0")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input = c->input(1);
      if (!c->RankKnown(input)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      int64 rank = c->Rank(input);
      if (rank == 0) {
        return errors::InvalidArgument(
            "Scalar cannot be used in ReduceScatter communications.");
      }
      std::vector<shape_inference::DimensionHandle> dims(rank);
      dims[0] = c->UnknownDim();
      for (int32 i = 1; i < rank; ++i) {
        dims[i] = c->Dim(input, i);
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .SetIsStateful()
    .Doc(R"doc(
ReduceScatter using a NCCL communicator.

output: A reduced and then scattered tensor.
handle: Handle of a NCCL communicator.
input: A tensor to reduce and scatter.
size: Total number of devices in the communicator.
rank: Index of current device in the communicator.
reduce_op: Reduce ops: 0 for SUM, 1 for PROD, 2 for MAX, 3 for MIN.
)doc");

#if GOOGLE_CUDA
template <typename T>
class NcclCommunicatorReduceScatterOp : public NcclCommunicatorAsyncOp {
 public:
  explicit NcclCommunicatorReduceScatterOp(OpKernelConstruction* ctx)
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

    // Allocate output with a 1/N size buffer of input tensor, here N is the
    // communicator size which indicates the device amounts.
    TensorShape out_shape(input->shape());
    OP_REQUIRES_ASYNC(
        ctx, out_shape.dims() > 0 && out_shape.dim_size(0) % comm->size() == 0,
        errors::InvalidArgument("Tensor cannot be scattered to ", comm->size(),
                                " devices with shape ",
                                out_shape.DebugString()),
        done);
    out_shape.set_dim(0, out_shape.dim_size(0) / comm->size());
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, out_shape, &output),
                         done);
    stream()->ThenWaitFor(ctx->op_device_context()->stream());

    VLOG(1) << comm->DebugString() << " [" << name() << "] [ReduceScatter]";
    OP_REQUIRES_OK_ASYNC(
        ctx, comm->ReduceScatter<T>(output, input, reduce_op_, cuda_stream()),
        done);
    done();
  }

 private:
  int reduce_op_;
};

#define REGISTER_KERNEL(TYPE)                                      \
  REGISTER_KERNEL_BUILDER(Name("EplNcclCommunicatorReduceScatter") \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<TYPE>("T"),          \
                          NcclCommunicatorReduceScatterOp<TYPE>);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

}  // namespace communicators
}  // namespace tensorflow
