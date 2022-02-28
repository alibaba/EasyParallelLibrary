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
#include <vector>

#include "communicators/nccl_communicator.h"

namespace tensorflow {
namespace communicators {

REGISTER_OP("EplNcclCommunicatorAllToAll")
    .Output("output: T")
    .Input("handle: resource")
    .Input("input: T")
    .Attr("rank: int >= 0 = 0")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    })
    .Doc(R"doc(
AllToAll using a NCCL communicator.

output: Exchanged tensor for each device.
handle: Handle of a NCCL communicator.
input: Tensor to be exchanged bettween each device.
rank: Index of current device in the communicator.
)doc");

#if GOOGLE_CUDA
template <typename T>
class NcclCommunicatorAllToAllOp : public NcclCommunicatorAsyncOp {
 public:
  explicit NcclCommunicatorAllToAllOp(OpKernelConstruction* ctx)
      : NcclCommunicatorAsyncOp(ctx) {}

  virtual void ComputeAsyncWithCommunicator(NcclCommunicator* comm,
                                            OpKernelContext* ctx,
                                            DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);

    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, input->shape(), &output),
                         done);
    se::Event* outputs_allocated = RecordEventOnTensorStream(ctx);
    WaitThenDeleteEventOnStream(outputs_allocated);

    VLOG(1) << comm->DebugString() << " [" << name() << "] [AllToAll]";
    OP_REQUIRES_OK_ASYNC(ctx, comm->AllToAll<T>(output, input, cuda_stream()),
                         done);

    done();
  }
};

#define REGISTER_KERNEL(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(Name("EplNcclCommunicatorAllToAll") \
                              .Device(DEVICE_GPU)             \
                              .TypeConstraint<TYPE>("T"),     \
                          NcclCommunicatorAllToAllOp<TYPE>);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

REGISTER_OP("EplNcclCommunicatorAllToAllv")
    .Output("outputs: P * T")
    .Input("handle: resource")
    .Input("inputs: P * T")
    .Attr("common_shape: shape = {}")
    .Attr("rank: int >= 0 = 0")
    .Attr("P: int >= 1 = 1")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      PartialTensorShape common_shape;
      TF_RETURN_IF_ERROR(c->GetAttr("common_shape", &common_shape));
      shape_inference::ShapeHandle shape;
      TF_RETURN_IF_ERROR(
          c->MakeShapeFromPartialTensorShape(common_shape, &shape));
      TF_RETURN_IF_ERROR(c->Concatenate(
          c->Vector(shape_inference::InferenceContext::kUnknownDim), shape,
          &shape));
      for (int64 dim = 0; dim < c->num_outputs(); ++dim) {
        c->set_output(dim, shape);
      }
      return Status::OK();
    })
    .Doc(R"doc(
AllToAllv using a NCCL communicator.

outputs: Rotated tensors for each device.
handle: Handle of a NCCL communicator.
inputs: Tensors to rotate for each device.
rank: Index of current device in the communicator.
)doc");

#if GOOGLE_CUDA
template <typename T>
class NcclCommunicatorAllToAllvOp : public NcclCommunicatorAsyncOp {
 public:
  explicit NcclCommunicatorAllToAllvOp(OpKernelConstruction* ctx)
      : NcclCommunicatorAsyncOp(ctx) {
    PartialTensorShape common_shape;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("common_shape", &common_shape));
    PartialTensorShape({1})
        .Concatenate(common_shape)
        .AsTensorShape(&output_shape_);
    common_shape_size_ = 1;
    for (int64 dim = 1; dim < output_shape_.dims(); ++dim) {
      common_shape_size_ *= output_shape_.dim_size(dim);
    }
  }

  virtual void ComputeAsyncWithCommunicator(NcclCommunicator* comm,
                                            OpKernelContext* ctx,
                                            DoneCallback done) override {
    // Get inputs.
    OpInputList inputs;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input_list("inputs", &inputs), done);

    // Collect sizes of all inputs across devices.
    std::vector<int64> dim0_sizes;
    {
      AllocatorAttributes local_sizes_alloc_attrs;
      local_sizes_alloc_attrs.set_on_host(true);
      local_sizes_alloc_attrs.set_gpu_compatible(true);
      Tensor local_sizes;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(DT_INT64, TensorShape({comm->size()}),
                             &local_sizes, local_sizes_alloc_attrs),
          done);
      for (int i = 0; i < comm->size(); ++i) {
        local_sizes.flat<int64>()(i) = inputs[i].NumElements();
      }
      Tensor local_sizes_on_gpu;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(DT_INT64, TensorShape({comm->size()}),
                             &local_sizes_on_gpu, ctx->output_alloc_attr(0)),
          done);
      Tensor all_sizes_on_gpu;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(DT_INT64,
                             TensorShape({comm->size() * comm->size()}),
                             &all_sizes_on_gpu, ctx->output_alloc_attr(0)),
          done);
      AllocatorAttributes all_sizes_alloc_attrs;
      all_sizes_alloc_attrs.set_on_host(true);
      Tensor all_sizes;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(DT_INT64,
                             TensorShape({comm->size() * comm->size()}),
                             &all_sizes, all_sizes_alloc_attrs),
          done);
      se::Event* scratch_allocated = RecordEventOnTensorStream(ctx);
      WaitThenDeleteEventOnStream(scratch_allocated);

      se::DeviceMemoryBase local_sizes_on_gpu_ptr(
          const_cast<char*>(local_sizes_on_gpu.tensor_data().data()),
          local_sizes_on_gpu.TotalBytes());
      stream()->ThenMemcpy(&local_sizes_on_gpu_ptr,
                           local_sizes.tensor_data().data(),
                           local_sizes.TotalBytes());
      VLOG(1) << comm->DebugString() << " [" << name() << "] [AllGather]";
      OP_REQUIRES_OK_ASYNC(
          ctx,
          comm->AllGather<int64>(&all_sizes_on_gpu, &local_sizes_on_gpu,
                                 CudaStream(stream())),
          done);
      se::DeviceMemoryBase all_sizes_on_gpu_ptr(
          const_cast<char*>(all_sizes_on_gpu.tensor_data().data()),
          all_sizes_on_gpu.TotalBytes());
      stream()->ThenMemcpy(const_cast<char*>(all_sizes.tensor_data().data()),
                           all_sizes_on_gpu_ptr, all_sizes.TotalBytes());
      stream()->BlockHostUntilDone();
      for (int64 i = 0; i < all_sizes.NumElements(); ++i) {
        int64 output_size = all_sizes.flat<int64>()(i);
        OP_REQUIRES_ASYNC(ctx, output_size % common_shape_size_ == 0,
                          errors::InvalidArgument(
                              "common_shape is not compatible with inputs"),
                          done);
        dim0_sizes.push_back(output_size / common_shape_size_);
      }
    }

    // Redirect symmetric input to output.
    ctx->set_output(comm->rank(), inputs[comm->rank()]);

    // Allocate outputs for non-symmetric inputs.
    for (int i = 0; i < comm->size(); ++i) {
      if (i == comm->rank()) {
        continue;
      }
      TensorShape output_shape(output_shape_);
      output_shape.set_dim(0, dim0_sizes[comm->size() * i + comm->rank()]);
      Tensor* output;
      OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(i, output_shape, &output),
                           done);
    }
    se::Event* outputs_allocated = RecordEventOnTensorStream(ctx);
    WaitThenDeleteEventOnStream(outputs_allocated);

    // AllToAllv for partial inputs.
    OpOutputList outputs;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->output_list("outputs", &outputs), done);
    VLOG(1) << comm->DebugString() << " [" << name() << "] [AllToAllv]";
    OP_REQUIRES_OK_ASYNC(
        ctx, comm->AllToAllv<T>(outputs, inputs, cuda_stream()), done);

    // done.
    done();
  }

 private:
  TensorShape output_shape_;
  int64 common_shape_size_;
};

#define REGISTER_KERNEL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("EplNcclCommunicatorAllToAllv") \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<TYPE>("T"),      \
                          NcclCommunicatorAllToAllvOp<TYPE>);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

}  // namespace communicators
}  // namespace tensorflow
