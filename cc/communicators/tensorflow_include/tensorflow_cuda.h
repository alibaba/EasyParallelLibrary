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
#ifndef TENSORFLOW_CUDA_H_
#define TENSORFLOW_CUDA_H_

#if GOOGLE_CUDA
#include "communicators/tensorflow_include/tensorflow.h"

namespace tensorflow {
namespace communicators {

inline Status CudaErrorToStatus(cudaError_t rc) {
  if (!TF_PREDICT_TRUE(cudaSuccess == rc)) {
    return errors::Internal(cudaGetErrorString(rc));
  }
  return Status::OK();
}

class CudaStream {
 public:
  CudaStream(se::Stream* stream) {
#if (TF_MAJOR_VERSION * 1000L + TF_MINOR_VERSION) <= 1010L
    stream_ = reinterpret_cast<cudaStream_t*>(
        stream->implementation()->CudaStreamMemberHack());
#else
    stream_ = reinterpret_cast<cudaStream_t*>(
        stream->implementation()->GpuStreamMemberHack());
#endif
  }

  cudaStream_t get() { return *stream_; }
  Status Wait() { return CudaErrorToStatus(cudaStreamSynchronize(*stream_)); }

 private:
  cudaStream_t* stream_;
};

class CudaStreamAsyncOpKernel : public AsyncOpKernel {
 public:
  explicit CudaStreamAsyncOpKernel(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx) {
    string thread_pool_name("stream_executor_async_op_thread_");
    string op_name(name());
    for (size_t i = 0; i < op_name.size(); ++i) {
      const char ch = op_name[i];
      if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
          (ch >= '0' && ch <= '9') || ch == '_' || ch == '-') {
        thread_pool_name += ch;
      } else {
        thread_pool_name += '_';
      }
    }
    thread_pool_.reset(new thread::ThreadPool(
        ctx->env(), ThreadOptions(), thread_pool_name, 1 /* num_threads */,
        false /* low_latency_hint */));
  }

  se::Stream* stream() { return se_stream_.get(); }

  CudaStream cuda_stream() { return CudaStream(se_stream_.get()); }

  void ThenExecute(OpKernelContext* ctx, AsyncOpKernel::DoneCallback func) {
    ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream(), func);
  }

  se::Event* RecordEventOnTensorStream(OpKernelContext* ctx) {
    se::Event* ev = new se::Event(ctx->op_device_context()->stream()->parent());
    ev->Init();
    ctx->op_device_context()->stream()->ThenRecordEvent(ev);
    return ev;
  }

  se::Event* RecordEventOnStream(OpKernelContext* ctx) {
    se::Event* ev = new se::Event(ctx->op_device_context()->stream()->parent());
    ev->Init();
    stream()->ThenRecordEvent(ev);
    return ev;
  }

  void WaitThenDeleteEventOnTensorStream(se::Event* ev, OpKernelContext* ctx) {
    ctx->op_device_context()->stream()->ThenWaitFor(ev);
    delete ev;
  }

  void WaitThenDeleteEventOnStream(se::Event* ev) {
    stream()->ThenWaitFor(ev);
    delete ev;
  }

  virtual void ComputeAsyncInternal(OpKernelContext* ctx,
                                    AsyncOpKernel::DoneCallback done) = 0;

  void ComputeAsync(OpKernelContext* ctx,
                    AsyncOpKernel::DoneCallback done) override {
    if (!se_stream_) {
      se_stream_.reset(
          new se::Stream(ctx->op_device_context()->stream()->parent()));
      // NOTE(zycao): CU_STREAM_NON_BLOCKING was not supported yet.
      // Using this argument for non-blocking by default stream.
      se_stream_->Init();
    }
    se::Event* input_event = RecordEventOnTensorStream(ctx);
    int device_id;
    cudaGetDevice(&device_id);
    auto wrapped_done = [this, ctx, done]() {
      se::Event* output_event = RecordEventOnStream(ctx);
      ThenExecute(ctx, [this, ctx, output_event, done]() {
        WaitThenDeleteEventOnTensorStream(output_event, ctx);
        done();
      });
    };
    thread_pool_->Schedule([device_id, this, ctx, input_event, wrapped_done]() {
      cudaSetDevice(device_id);
      WaitThenDeleteEventOnStream(input_event);
      se::cuda::ScopedActivateExecutorContext context(se_stream_->parent());
      ComputeAsyncInternal(ctx, wrapped_done);
    });
  }

 private:
  std::unique_ptr<se::Stream> se_stream_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
};

class CudaStreamOpKernel : public OpKernel {
 public:
  explicit CudaStreamOpKernel(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  se::Stream* stream() { return se_stream_.get(); }

  CudaStream cuda_stream() { return CudaStream(se_stream_.get()); }

  void ThenExecute(OpKernelContext* ctx, AsyncOpKernel::DoneCallback func) {
    ctx->device()->tensorflow_gpu_device_info()->event_mgr->ThenExecute(
        stream(), func);
  }

  se::Event* RecordEventOnTensorStream(OpKernelContext* ctx) {
    se::Event* ev = new se::Event(ctx->op_device_context()->stream()->parent());
    ev->Init();
    ctx->op_device_context()->stream()->ThenRecordEvent(ev);
    return ev;
  }

  se::Event* RecordEventOnStream(OpKernelContext* ctx) {
    se::Event* ev = new se::Event(ctx->op_device_context()->stream()->parent());
    ev->Init();
    stream()->ThenRecordEvent(ev);
    return ev;
  }

  void WaitThenDeleteEventOnTensorStream(se::Event* ev, OpKernelContext* ctx) {
    ctx->op_device_context()->stream()->ThenWaitFor(ev);
    delete ev;
  }

  void WaitThenDeleteEventOnStream(se::Event* ev) {
    stream()->ThenWaitFor(ev);
    delete ev;
  }

  virtual void ComputeInternal(OpKernelContext* ctx) = 0;

  void Compute(OpKernelContext* ctx) override {
    if (!se_stream_) {
      se_stream_.reset(
          new se::Stream(ctx->op_device_context()->stream()->parent()));
      // NOTE(zycao): CU_STREAM_NON_BLOCKING was not supported yet.
      // Using this argument for non-blocking by default stream.
      se_stream_->Init();
    }
    se::Event* input_event = RecordEventOnTensorStream(ctx);
    WaitThenDeleteEventOnStream(input_event);
    se::cuda::ScopedActivateExecutorContext context(se_stream_->parent());
    ComputeInternal(ctx);
    se::Event* output_event = RecordEventOnStream(ctx);
    WaitThenDeleteEventOnTensorStream(output_event, ctx);
  }

 private:
  std::unique_ptr<se::Stream> se_stream_;
};

}  // namespace communicators
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_CUDA_H_
