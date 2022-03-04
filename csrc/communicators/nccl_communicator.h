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
#ifndef NCCL_COMMUNICATOR_H_
#define NCCL_COMMUNICATOR_H_

#include "communicators/tensorflow_include/tensorflow.h"
#include "communicators/tensorflow_include/tensorflow_nccl.h"

#if GOOGLE_CUDA

namespace tensorflow {
namespace communicators {

class NcclCommunicator : public ResourceBase {
 public:
  NcclCommunicator() {}

  ~NcclCommunicator() {}

  string DebugString() TF_RESOURCE_DEBUG_STRING_CONST override {
    return debug_string_;
  }

  Status Create(const string& id, const string& shared_name, int size,
                int rank) {
    TF_RETURN_IF_ERROR(comm_.Create(id, size, rank));
    debug_string_ =
        strings::StrCat("NcclCommunicator(name=", shared_name,
                        ", size=", comm_.size(), ", rank=", comm_.rank(), ")");

    return Status::OK();
  }

  int rank() const { return comm_.rank(); }

  int size() const { return comm_.size(); }

  template <typename T>
  Status AllGather(Tensor* output, const Tensor* input, CudaStream stream) {
    void* recvbuf = const_cast<char*>(output->tensor_data().data());

    return comm_.AllGather<T>(input->tensor_data().data(), recvbuf,
                              input->NumElements(), stream);
  }

  template <typename T>
  Status AllGatherv(Tensor* output, const Tensor* input,
                    const int64* input_sizes, CudaStream stream) {
    void* recvbuf = const_cast<char*>(output->tensor_data().data());

    return comm_.AllGatherv<T>(input->tensor_data().data(), recvbuf,
                               input_sizes, stream);
  }

  template <typename T>
  Status AllReduce(Tensor* output, const Tensor* input, const int reduce_op,
                   CudaStream stream) {
    const void* sendbuf = input->tensor_data().data();
    void* recvbuf = const_cast<char*>(output->tensor_data().data());

    return comm_.AllReduce<T>(sendbuf, recvbuf, input->NumElements(), reduce_op,
                              stream);
  }

  template <typename T>
  Status Broadcast(Tensor* output, const Tensor* input, const int root_rank,
                   CudaStream stream) {
    void* recvbuf = const_cast<char*>(output->tensor_data().data());

    return comm_.Broadcast<T>(input->tensor_data().data(), recvbuf,
                              input->NumElements(), root_rank, stream);
  }

  template <typename T>
  Status ReduceScatter(Tensor* output, const Tensor* input, const int reduce_op,
                       CudaStream stream) {
    void* recvbuf = const_cast<char*>(output->tensor_data().data());

    return comm_.ReduceScatter<T>(input->tensor_data().data(), recvbuf,
                                  output->NumElements(), reduce_op, stream);
  }

  template <typename T>
  Status Reduce(Tensor* output, const Tensor* input, const int reduce_op,
                const int root_rank, CudaStream stream) {
    void* recvbuf = const_cast<char*>(output->tensor_data().data());

    return comm_.Reduce<T>(input->tensor_data().data(), recvbuf,
                           input->NumElements(), reduce_op, root_rank, stream);
  }

  template <typename T>
  Status AllToAll(Tensor* output, const Tensor* input, CudaStream stream) {
    const void* sendbuf = input->tensor_data().data();
    void* recvbuf = const_cast<char*>(output->tensor_data().data());
    return comm_.AllToAll<T>(sendbuf, recvbuf, input->NumElements(), stream);
  }

  template <typename T>
  Status AllToAllv(OpOutputList& recv_tensors, const OpInputList& send_tensors,
                   CudaStream stream) {
    if (TF_PREDICT_FALSE(send_tensors.size() != size() ||
                         recv_tensors.size() != size())) {
      return errors::InvalidArgument(
          "Size of send_tensors and recv_tensors must be same to size of the "
          "communicator");
    }
    std::vector<const void*> sendbufs(send_tensors.size(), nullptr);
    std::vector<void*> recvbufs(recv_tensors.size(), nullptr);
    std::vector<size_t> sendcounts(send_tensors.size(), 0);
    std::vector<size_t> recvcounts(recv_tensors.size(), 0);
    for (int i = 0; i < send_tensors.size(); ++i) {
      if (rank() != i) {
        sendbufs[i] = send_tensors[i].tensor_data().data();
        recvbufs[i] = const_cast<char*>(recv_tensors[i]->tensor_data().data());
        sendcounts[i] = send_tensors[i].NumElements();
        recvcounts[i] = recv_tensors[i]->NumElements();
      }
    }
    return comm_.AllToAllv<T>(sendbufs, recvbufs, sendcounts, recvcounts,
                              stream);
  }

 private:
  NcclCommWrapper comm_;
  string debug_string_;
};

class NcclCommunicatorAsyncOp : public CudaStreamAsyncOpKernel {
 public:
  explicit NcclCommunicatorAsyncOp(OpKernelConstruction* ctx)
      : CudaStreamAsyncOpKernel(ctx) {}

  virtual void ComputeAsyncWithCommunicator(NcclCommunicator* comm,
                                            OpKernelContext* ctx,
                                            DoneCallback done) = 0;

  virtual void ComputeAsyncInternal(OpKernelContext* ctx,
                                    DoneCallback done) override {
    NcclCommunicator* comm = nullptr;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &comm), done);

    ComputeAsyncWithCommunicator(comm, ctx, done);
  };
};

}  // namespace communicators
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // NCCL_COMMUNICATOR_H_
