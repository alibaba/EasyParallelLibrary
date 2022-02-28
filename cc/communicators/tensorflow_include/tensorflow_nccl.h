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
#ifndef TENSORFLOW_NCCL_H_
#define TENSORFLOW_NCCL_H_

#if GOOGLE_CUDA

#include <nccl.h>

#include "communicators/tensorflow_include/tensorflow_cuda.h"

namespace tensorflow {
namespace communicators {

template <typename TYPE>
struct DataTypeToNcclEnum {
  static constexpr ncclDataType_t value = ncclFloat;
};

template <ncclDataType_t VALUE>
struct NcclEnumToDataType {
  typedef float Type;
};

#define MATCH_TYPE_AND_NCCL_ENUM(TYPE, ENUM)      \
  template <>                                     \
  struct DataTypeToNcclEnum<TYPE> {               \
    static constexpr ncclDataType_t value = ENUM; \
  };                                              \
  template <>                                     \
  struct NcclEnumToDataType<ENUM> {               \
    typedef TYPE Type;                            \
  }

MATCH_TYPE_AND_NCCL_ENUM(int8, ncclInt8);
MATCH_TYPE_AND_NCCL_ENUM(uint8, ncclUint8);
MATCH_TYPE_AND_NCCL_ENUM(int32, ncclInt32);
MATCH_TYPE_AND_NCCL_ENUM(uint32, ncclUint32);
MATCH_TYPE_AND_NCCL_ENUM(int64, ncclInt64);
MATCH_TYPE_AND_NCCL_ENUM(uint64, ncclUint64);
MATCH_TYPE_AND_NCCL_ENUM(Eigen::half, ncclFloat16);
MATCH_TYPE_AND_NCCL_ENUM(float, ncclFloat32);
MATCH_TYPE_AND_NCCL_ENUM(double, ncclFloat64);

#define TF_CALL_NCCL_TYPES(m)                                             \
  TF_CALL_int8(m) TF_CALL_uint8(m) TF_CALL_int32(m) TF_CALL_uint32(m)     \
      TF_CALL_int64(m) TF_CALL_uint64(m) TF_CALL_half(m) TF_CALL_float(m) \
          TF_CALL_double(m)

inline Status NcclErrorToStatus(ncclResult_t rc) {
  if (!TF_PREDICT_TRUE(ncclSuccess == rc)) {
    return errors::Internal(ncclGetErrorString(rc));
  }
  return Status::OK();
}

class NcclGroup {
 public:
  NcclGroup() { ncclGroupStart(); }
  ~NcclGroup() { ncclGroupEnd(); }
};

class NcclCommWrapper {
 public:
  NcclCommWrapper() : size_(1), rank_(0), init_(false) {}

  ~NcclCommWrapper() {
    if (init_) {
      Destroy();
    }
  }

  int size() const { return size_; }

  int rank() const { return rank_; }

  Status Create(const string& nccl_id_str, const int size, const int rank) {
    if (!TF_PREDICT_TRUE(nccl_id_str.size() == NCCL_UNIQUE_ID_BYTES)) {
      return errors::InvalidArgument(
          strings::StrCat("NCCL ID ", nccl_id_str.c_str(), " is invalid."));
    }

    if (!TF_PREDICT_TRUE(0 <= rank && rank < size)) {
      return errors::InvalidArgument(strings::StrCat(
          "NCCL rank ", rank, " or size ", size, " is invalid."));
    }

    size_ = size;
    rank_ = rank;

    ncclUniqueId nccl_id;
    memcpy(nccl_id.internal, &nccl_id_str[0], NCCL_UNIQUE_ID_BYTES);

    TF_RETURN_IF_ERROR(
        NcclErrorToStatus(ncclCommInitRank(&comm_, size_, nccl_id, rank_)));

    init_ = true;
    return Status::OK();
  }

  Status Destroy() {
    TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclCommDestroy(comm_)));
    init_ = false;
    return Status::OK();
  }

  Status Abort() {
    TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclCommAbort(comm_)));
    init_ = false;
    return Status::OK();
  }

  Status UserRank(int* rank) {
    return NcclErrorToStatus(ncclCommUserRank(comm_, rank));
  }

  template <typename T>
  Status Reduce(const void* sendbuf, void* recvbuf, const size_t count,
                const int reduce_op, const int root_rank, CudaStream stream) {
    return NcclErrorToStatus(ncclReduce(
        sendbuf, recvbuf, count, DataTypeToNcclEnum<T>::value,
        static_cast<ncclRedOp_t>(reduce_op), root_rank, comm_, stream.get()));
  }

  template <typename T>
  Status Broadcast(const void* sendbuf, void* recvbuf, const size_t count,
                   const int root_rank, CudaStream stream) {
    return NcclErrorToStatus(ncclBroadcast(sendbuf, recvbuf, count,
                                           DataTypeToNcclEnum<T>::value,
                                           root_rank, comm_, stream.get()));
  }

  template <typename T>
  Status AllReduce(const void* sendbuf, void* recvbuf, const size_t count,
                   const int reduce_op, CudaStream stream) {
    return NcclErrorToStatus(ncclAllReduce(
        sendbuf, recvbuf, count, DataTypeToNcclEnum<T>::value,
        static_cast<ncclRedOp_t>(reduce_op), comm_, stream.get()));
  }

  template <typename T>
  Status ReduceScatter(const void* sendbuf, void* recvbuf, const size_t count,
                       const int reduce_op, CudaStream stream) {
    return NcclErrorToStatus(ncclReduceScatter(
        sendbuf, recvbuf, count, DataTypeToNcclEnum<T>::value,
        static_cast<ncclRedOp_t>(reduce_op), comm_, stream.get()));
  }

  template <typename T>
  Status AllGather(const void* sendbuf, void* recvbuf, const size_t count,
                   CudaStream stream) {
    return NcclErrorToStatus(ncclAllGather(sendbuf, recvbuf, count,
                                           DataTypeToNcclEnum<T>::value, comm_,
                                           stream.get()));
  }

  template <typename T>
  Status AllGatherv(const void* sendbuf, void* recvbuf, const int64* countbuf,
                    CudaStream stream) {
    T* typed_recvbuf = reinterpret_cast<T*>(recvbuf);
    size_t offset = 0;
    ncclGroupStart();
    for (int i = 0; i < size_; ++i) {
      TF_RETURN_IF_ERROR(NcclErrorToStatus(ncclBroadcast(
          sendbuf, reinterpret_cast<void*>(typed_recvbuf + offset), countbuf[i],
          DataTypeToNcclEnum<T>::value, i, comm_, stream.get())));
      offset += countbuf[i];
    }
    ncclGroupEnd();
    return Status::OK();
  }

  template <typename T>
  Status AllToAll(const void* sendbuf, void* recvbuf, const size_t count,
                  CudaStream stream) {
#if NCCL_VERSION_CODE >= 2700
    ncclGroupStart();
    assert(size_ != 0);
    size_t segment = count / size_;

    for (int i = 0; i < size_; ++i) {
      TF_RETURN_IF_ERROR(NcclErrorToStatus(
          ncclSend(static_cast<const T*>(sendbuf) + i * segment, segment,
                   DataTypeToNcclEnum<T>::value, i, comm_, stream.get())));
      TF_RETURN_IF_ERROR(NcclErrorToStatus(
          ncclRecv(static_cast<T*>(recvbuf) + i * segment, segment,
                   DataTypeToNcclEnum<T>::value, i, comm_, stream.get())));
    }
    ncclGroupEnd();
    return Status::OK();
#else
    return errors::Unimplemented("AllToAll not supported in NCCL < 2.7");
#endif
  }

  template <typename T>
  Status AllToAllv(std::vector<const void*>& sendbufs,
                   std::vector<void*>& recvbufs,
                   std::vector<size_t>& sendcounts,
                   std::vector<size_t>& recvcounts, CudaStream stream) {
#if NCCL_VERSION_CODE >= 2700
    ncclGroupStart();
    for (int i = 0; i < size_; ++i) {
      if (rank_ == i) {
        if (sendbufs[i] == nullptr && recvbufs[i] == nullptr) {
          // senbufs[rank_] has been redirected to recvbufs[rank_].
          continue;
        } else if (sendbufs[i] == nullptr || recvbufs[i] == nullptr) {
          return errors::InvalidArgument(
              "sendbufs and recvbufs must both be nullptr at same rank.");
        }
      }
      TF_RETURN_IF_ERROR(NcclErrorToStatus(
          ncclSend(sendbufs[i], sendcounts[i], DataTypeToNcclEnum<T>::value, i,
                   comm_, stream.get())));
      TF_RETURN_IF_ERROR(NcclErrorToStatus(
          ncclRecv(recvbufs[i], recvcounts[i], DataTypeToNcclEnum<T>::value, i,
                   comm_, stream.get())));
    }
    ncclGroupEnd();
    return Status::OK();
#else
    return errors::Unimplemented("AllToAllv not supported in NCCL < 2.7");
#endif
  }

  template <typename T>
  Status AllToAllv(const void* sendbuf, void* recvbuf, const size_t* countbuf,
                   CudaStream stream) {
#if NCCL_VERSION_CODE >= 2700
    ncclGroupStart();
    size_t sendoffset = 0;
    size_t sendsize = 0;
    size_t recvoffset = 0;
    size_t recvsize = 0;
    for (int i = 0; i < size_; ++i) {
      sendsize = countbuf[size_ * rank_ + i];
      TF_RETURN_IF_ERROR(NcclErrorToStatus(
          ncclSend(static_cast<const char*>(sendbuf) + sendoffset, sendsize,
                   DataTypeToNcclEnum<T>::value, i, comm_, stream.get())));
      sendoffset += sendsize;
      recvsize = countbuf[size_ * i + rank_];
      TF_RETURN_IF_ERROR(NcclErrorToStatus(
          ncclRecv(static_cast<char*>(recvbuf) + recvoffset, recvsize,
                   DataTypeToNcclEnum<T>::value, i, comm_, stream.get())));
      recvoffset += recvsize;
    }
    ncclGroupEnd();
    return Status::OK();
#else
    return errors::Unimplemented("AllToAllv not supported in NCCL < 2.7");
#endif
  }

  template <typename T>
  Status BatchAllToAllv(std::vector<const void*>& sendbufs,
                        std::vector<void*>& recvbufs,
                        std::vector<size_t>& sendcounts,
                        std::vector<size_t>& recvcounts, size_t batch_size,
                        CudaStream stream) {
#if NCCL_VERSION_CODE >= 2700
    ncclDataType_t dtype = DataTypeToNcclEnum<T>::value;
    for (size_t k = 0; k < batch_size; ++k) {
      ncclGroupStart();
      for (int i = 0; i < size_; ++i) {
        const void* sendbuf = sendbufs[size_ * k + i];
        void* recvbuf = recvbufs[size_ * k + i];
        size_t sendcount = sendcounts[size_ * k + i];
        size_t recvcount = recvcounts[size_ * k + i];
        if (rank_ == i) {
          if (sendbuf == nullptr && recvbuf == nullptr) {
            continue;
          } else if (sendbuf == nullptr || recvbuf == nullptr) {
            return errors::InvalidArgument(
                "sendbufs and recvbufs must both be nullptr at same rank.");
          }
        }
        TF_RETURN_IF_ERROR(NcclErrorToStatus(
            ncclSend(sendbuf, sendcount, dtype, i, comm_, stream.get())));
        TF_RETURN_IF_ERROR(NcclErrorToStatus(
            ncclRecv(recvbuf, recvcount, dtype, i, comm_, stream.get())));
      }
      ncclGroupEnd();
    }
    return Status::OK();
#else
    return errors::Unimplemented("BatchAllToAllv not supported in NCCL < 2.7");
#endif
  }

 private:
  ncclComm_t comm_;
  int size_;
  int rank_;
  bool init_;
};

}  // namespace communicators
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
#endif  // TENSORFLOW_NCCL_H_
