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

REGISTER_OP("EplNcclCommunicatorAllGather")
    .Output("output: T")
    .Input("handle: resource")
    .Input("input: T")
    .Attr("size: int >= 1 = 1")
    .Attr("rank: int >= 0 = 0")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input = c->input(1);
      if (!c->RankKnown(input)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      int64 rank = c->Rank(input);
      if (rank == 0) {
        rank = 1;
      }  // For Allgather of scalar tensors.
      std::vector<shape_inference::DimensionHandle> dims(rank);
      dims[0] = c->UnknownDim();
      for (int32 i = 1; i < rank; ++i) {
        dims[i] = c->Dim(input, i);
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
AllGather using a NCCL communicator.

output: A gathered tensor.
handle: Handle of a NCCL communicator.
input: A tensor to gather.
size: Total number of devices in the communicator.
rank: Index of current device in the communicator.
)doc");

#if GOOGLE_CUDA
template <typename T>
class NcclCommunicatorAllGatherOp : public NcclCommunicatorAsyncOp {
 public:
  explicit NcclCommunicatorAllGatherOp(OpKernelConstruction* ctx)
      : NcclCommunicatorAsyncOp(ctx) {}

  void ComputeAsyncWithCommunicator(NcclCommunicator* comm,
                                    OpKernelContext* ctx,
                                    DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);

    // Allocate output with a N times larger buffer than input, here N is the
    // communicator size which indicates the device amounts.
    TensorShape out_shape(input->shape());
    if (out_shape.dims() == 0) {
      out_shape.AddDim(comm->size());
    } else {
      out_shape.set_dim(0, out_shape.dim_size(0) * comm->size());
    }
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, out_shape, &output),
                         done);
    se::Event* outputs_allocated = RecordEventOnTensorStream(ctx);
    WaitThenDeleteEventOnStream(outputs_allocated);

    VLOG(1) << comm->DebugString() << " [" << name() << "] [AllGather]";
    OP_REQUIRES_OK_ASYNC(ctx, comm->AllGather<T>(output, input, cuda_stream()),
                         done);
    done();
  }
};

#define REGISTER_KERNEL(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(Name("EplNcclCommunicatorAllGather") \
                              .Device(DEVICE_GPU)              \
                              .TypeConstraint<TYPE>("T"),      \
                          NcclCommunicatorAllGatherOp<TYPE>);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

REGISTER_OP("EplNcclCommunicatorAllGatherv")
    .Output("output: T")
    .Input("handle: resource")
    .Input("input: T")
    .Attr("size: int >= 1 = 1")
    .Attr("rank: int >= 0 = 0")
    .Attr("T: {int8, uint8, int32, uint32, int64, uint64, half, float, double}")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle input = c->input(1);
      if (!c->RankKnown(input)) {
        c->set_output(0, c->UnknownShape());
        return Status::OK();
      }

      int64 rank = c->Rank(input);
      if (rank == 0) {
        rank = 1;
      }  // For Allgatherv of scalar tensors.
      std::vector<shape_inference::DimensionHandle> dims(rank);
      dims[0] = c->UnknownDim();
      for (int32 i = 1; i < rank; ++i) {
        dims[i] = c->Dim(input, i);
      }
      c->set_output(0, c->MakeShape(dims));
      return Status::OK();
    })
    .Doc(R"doc(
AllGatherv using a NCCL communicator.

output: A gathered tensor.
handle: Handle of a NCCL communicator.
input: A tensor to gather.
size: Total number of devices in the communicator.
rank: Index of current device in the communicator.
)doc");

#if GOOGLE_CUDA
template <typename T>
class NcclCommunicatorAllGathervOp : public NcclCommunicatorAsyncOp {
 public:
  explicit NcclCommunicatorAllGathervOp(OpKernelConstruction* ctx)
      : NcclCommunicatorAsyncOp(ctx) {}

  void ComputeAsyncWithCommunicator(NcclCommunicator* comm,
                                    OpKernelContext* ctx,
                                    DoneCallback done) override {
    const Tensor* input;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("input", &input), done);

    // Collect sizes of all inputs across devices.
    Tensor all_sizes;
    {
      AllocatorAttributes local_sizes_alloc_attrs;
      local_sizes_alloc_attrs.set_on_host(true);
      local_sizes_alloc_attrs.set_gpu_compatible(true);
      Tensor local_sizes;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(DT_INT64, TensorShape({}), &local_sizes,
                             local_sizes_alloc_attrs),
          done);
      local_sizes.scalar<int64>()() = input->NumElements();
      Tensor local_sizes_on_gpu;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(DT_INT64, TensorShape({}), &local_sizes_on_gpu,
                             ctx->output_alloc_attr(0)),
          done);
      Tensor all_sizes_on_gpu;
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(DT_INT64, TensorShape({comm->size()}),
                             &all_sizes_on_gpu, ctx->output_alloc_attr(0)),
          done);
      AllocatorAttributes all_sizes_alloc_attrs;
      all_sizes_alloc_attrs.set_on_host(true);
      OP_REQUIRES_OK_ASYNC(
          ctx,
          ctx->allocate_temp(DT_INT64, TensorShape({comm->size()}), &all_sizes,
                             all_sizes_alloc_attrs),
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
    }

    // Now all Tensor sizes have been gathered, calculate the total size and
    // allocate the output Tensor for communication.
    int64 total_size = 0;
    bool is_same_size = true;
    for (int i = 0; i < comm->size(); ++i) {
      int64 per_elements = all_sizes.flat<int64>()(i);
      total_size += per_elements;
      if (is_same_size && per_elements != input->NumElements()) {
        is_same_size = false;
      }
    }
    TensorShape out_shape(input->shape());
    int64 sub_size = 1;
    for (int i = 1; i < out_shape.dims(); ++i) {
      sub_size *= out_shape.dim_size(i);
    }
    if (out_shape.dims() == 0) {
      out_shape.AddDim(total_size);
    } else {
      out_shape.set_dim(0, total_size / sub_size);
    }
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->allocate_output(0, out_shape, &output),
                         done);
    se::Event* outputs_allocated = RecordEventOnTensorStream(ctx);
    WaitThenDeleteEventOnStream(outputs_allocated);

    if (is_same_size) {
      // Directly call Allgather when num_elements are same in all ranks.
      VLOG(1) << comm->DebugString() << " [" << name() << "] [AllGather]";
      OP_REQUIRES_OK_ASYNC(
          ctx, comm->AllGather<T>(output, input, cuda_stream()), done);
      done();
      return;
    }
    const int64* all_sizes_ptr =
        reinterpret_cast<const int64*>(all_sizes.flat<int64>().data());
    VLOG(1) << comm->DebugString() << " [" << name() << "] [AllGatherv]";
    OP_REQUIRES_OK_ASYNC(
        ctx, comm->AllGatherv<T>(output, input, all_sizes_ptr, cuda_stream()),
        done);
    done();
  }
};

#define REGISTER_KERNEL(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(Name("EplNcclCommunicatorAllGatherv") \
                              .Device(DEVICE_GPU)               \
                              .TypeConstraint<TYPE>("T"),       \
                          NcclCommunicatorAllGathervOp<TYPE>);
TF_CALL_NCCL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif  // GOOGLE_CUDA

}  // namespace communicators
}  // namespace tensorflow
