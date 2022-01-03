/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_COMPUTATION_H_
#define NCCL_PRIMITIVES_COMPUTATION_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"

#define SPINS_BEFORE_CHECK_ABORT 1000000

// Unroll unconditionally the first send/recv since nsend/nrecv should be at
// least 1 if SEND/RECV is set.
#define FOR_SEND(func, ...) do { \
  if (SEND) { \
    /* Send to far first, then close */ \
    for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__); \
    func(0, ##__VA_ARGS__); \
  } \
} while (0)

#define FOR_RECV(func, ...) do { \
  if (RECV) { \
    /* Recv from close first, then far */ \
    func(0, ##__VA_ARGS__); \
    for (int i=1; i<NRECV && i<nrecv; i++) func(i, ##__VA_ARGS__); \
  } \
} while (0)

#define TYPE_PRIMS 0









#if TYPE_PRIMS == 0

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, int NRECV, int NSEND, typename TF16, class FUNCF16, typename TF32, class FUNCF32>
class ncclPrimitivesComputation {
 private:
  const int stepSize;
  const int tid;
  const int nthreads;
  const int wid;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;
  volatile uint64_t* recvConnTailPtr = NULL;
  uint64_t recvConnTail;
  uint64_t recvConnTailCache; // Cache last seen value

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  const TF16* recvDirectBuff[NRECV];
  TF16* sendDirectBuff[NSEND];
  const TF16* recvBuff[NRECV];
  TF16* sendBuff[NSEND];
  struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ const TF16* recvPtr(int i) { return ((const TF16*)recvBuff[i])+recvOffset(i); }
  inline __device__ TF16* sendPtr(int i) { return ((TF16*)sendBuff[i])+sendOffset(i); }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
  }
  inline __device__ void subBarrier() {
    asm volatile ("bar.sync 2, %0;" :: "r"(nthreads-WARP_SIZE));
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  inline __device__ void checkMismatch(struct ncclConnInfo* conn) {
    if (mismatch) {
      // In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch
      *(comm->fatalDevError) = ncclDevAssertedMismatch;
    } else if (conn && *conn->opCountRem > opCount) {
      mismatch += 1;
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort(int i, int send) {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *(comm->abortFlag);
      if (wid == i) checkMismatch(send ? sendConn : recvConn);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    spins = 0;
    mismatch = 0;
    if (sendConnHeadPtr) {
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + SLICESTEPS) {
        sendConnHeadCache = *sendConnHeadPtr;
        if (checkAbort(wid, 1)) break;
      }
      if (sendConnFifoPtr) {
        sendConnFifoPtr[sendConnHead%NCCL_STEPS] = nbytes;
      }
      sendConnHead += SLICESTEPS;
    }
  }

  inline __device__ void waitRecv() {
    spins = 0;
    mismatch = 0;
    if (recvConnTailPtr) {
      while (recvConnTailCache < recvConnTail + SLICESTEPS) {
        recvConnTailCache = *recvConnTailPtr;
        if (checkAbort(wid, 0)) break;
      }
      recvConnTail += SLICESTEPS;
    }
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += SLICESTEPS;
  }
  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += SLICESTEPS;
  }

  inline __device__ void incSend(int i) {
    sendStep[i] += SLICESTEPS;
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) *sendConnTailPtr = sendConnTail += SLICESTEPS;
  }

  template <int DIRECTRECV>
  inline __device__ const TF16* directRecvPtr(int i, ssize_t directOffset) {
    return DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : recvPtr(i);
  }

  template <int DIRECTSEND>
  inline __device__ TF16* directSendPtr(int i, ssize_t directOffset) {
    return DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : sendPtr(i);
  }

  template <int DIRECTRECV>
  inline __device__ int directRecvInc(int i, int directInc, int sliceInc) {
    return DIRECTRECV && recvDirectBuff[i] ? directInc : sliceInc;
  }

  template <int DIRECTSEND>
  inline __device__ int directSendInc(int i, int directInc, int sliceInc) {
    return DIRECTSEND && sendDirectBuff[i] ? directInc : sliceInc;
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST, int COMPUTE, int ALLGATHER_COMPUTE, typename T, class FUNC>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, T lr, T beta1, T beta2, T* m, T* v, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);
    // assert(COMPUTE == 0 && ALLGATHER_COMPUTE==0);
    const T* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    bool syncThread = tid >= nthreads-WARP_SIZE;

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (!syncThread) {
        if (SEND) waitSend(realSize*sizeof(T));
        if (RECV) waitRecv();
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              //ReduceOrCopyMultiComputation<UNROLL, FUNC, T, 1, 1, 1, NSEND, COMPUTE, ALLGATHER_COMPUTE>(tid, nthreads-WARP_SIZE, 1, srcs, nsend, dsts+1, realSize, lr, beta1, beta2, m, v);
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND>(tid, nthreads-WARP_SIZE, 1, srcs, nsend, dsts+1, realSize);
            }
          } else {
            //ReduceOrCopyMultiComputation<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, COMPUTE, ALLGATHER_COMPUTE>(tid, nthreads-WARP_SIZE, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize, lr, beta1, beta2, m, v);
            ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads-WARP_SIZE, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
          }
        }
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (syncThread) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
      dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST, int COMPUTE, int ALLGATHER_COMPUTE>
  inline __device__ void
  GenericOp2(const TF16* srcPtr, TF16* dstPtr, TF32* w, TF32 lr, TF32 beta1, TF32 beta2, TF32* m, TF32* v, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    const TF16* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    TF16* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    bool syncThread = tid >= nthreads-WARP_SIZE;

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (!syncThread) {
        if (SEND) waitSend(realSize*sizeof(TF16));
        if (RECV) waitRecv();
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              //assert(false);
              //ReduceOrCopyMultiComputation2<UNROLL, FUNCF16, TF16, TF32, 1, 1, 1, NSEND, COMPUTE, ALLGATHER_COMPUTE>(tid, nthreads-WARP_SIZE, 1, srcs, nsend, dsts+1, realSize, lr, beta1, beta2, m, v);
            }
          } else {
            ReduceOrCopyMultiComputation2<UNROLL, FUNCF16, TF16, TF32, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, COMPUTE, ALLGATHER_COMPUTE>(tid, nthreads-WARP_SIZE, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize, w+offset, lr, beta1, beta2, m+offset, v+offset);
          }
        }
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (syncThread) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
      dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i, TF16* directBuff) {
    recvBuff[i] = (const TF16*)conn->buff;
    recvStep[i] = conn->step;
    recvStep[i] = ROUNDUP(recvStep[i], SLICESPERCHUNK*SLICESTEPS);
    recvDirectBuff[i] = NULL;
    if (directBuff && (conn->direct & NCCL_DIRECT_GPU)) {
      recvDirectBuff[i] = directBuff;
      if (tid == 0) *conn->ptrExchange = directBuff;
    }
    if (wid == i) recvConn = conn;
    if (wid == i) recvConnTail = recvConnHead = recvStep[i]; // Make sure we set this after rounding up
    nrecv++;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= WARP_SIZE && tid < 2*WARP_SIZE && wid<nrecv) {
      recvConnTailPtr = recvConn->tail;
      recvConnTailCache = *recvConnTailPtr;
    }
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConnHeadPtr = recvConn->head;
      // Return credits in case we rounded up.
      *recvConnHeadPtr = recvConnHead;
      // Update opCount in case we skipped some operations
      *(recvConn->opCountLoc) = opCount;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i, TF16* directBuff) {
    sendBuff[i] = (TF16*)conn->buff;
    sendStep[i] = conn->step;
    sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);
    sendDirectBuff[i] = NULL;
    if (directBuff && (conn->direct & NCCL_DIRECT_GPU)) {
      void* volatile* ptr = conn->ptrExchange;
      while ((sendDirectBuff[i] = (TF16*)(*ptr)) == NULL);
      barrier();
      if (tid == 0) *ptr = NULL;
    }
    if (wid == i) sendConn = conn;
    if (wid == i) sendConnTail = sendConnHead = sendStep[i]; // Make sure we set this after rounding up
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnFifoPtr = sendConn->fifo;
      *(sendConn->opCountLoc) = opCount;
    }
    if (tid >= nthreads-WARP_SIZE && wid<nsend) {
      sendConnTailPtr = sendConn->tail;
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConn->step = recvConnHead;
      *(recvConn->opCountLoc) = opCount+1;
      __threadfence_system();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      sendConn->step = sendConnHead;
      *(sendConn->opCountLoc) = opCount+1;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitivesComputation(const int tid, const int nthreads, int* recvPeers, int* sendPeers, TF16* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize), opCount(opCount) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, directBuff);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i, directBuff);
    loadRecvSync();
    loadSendSync();
  }

  //TF16
  __device__ __forceinline__ void
  send(const TF16* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0, 0, 0, TF16, FUNCF16>(src, NULL, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, 0);
  }
  __device__ __forceinline__ void
  directSend(const TF16* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0, 0, 0, TF16, FUNCF16>(src, NULL, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, directOffset);
  }

  __device__ __forceinline__ void
  recv(TF16* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1, 0, 0, TF16, FUNCF16>(NULL, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, 0);
  }

    __device__ __forceinline__ void
  recvAllGatherCompute(TF16* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1, 0, 1, TF16, FUNCF16>(NULL, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, 0);
  }

  __device__ __forceinline__ void
  directRecv(TF16* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1, 0, 0, TF16, FUNCF16>(NULL, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, directOffset);
  }

  __device__ __forceinline__ void
  directRecvAllGatherCompute(TF16* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1, 0, 1, TF16, FUNCF16>(NULL, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, directOffset);
  }

  __device__ __forceinline__ void
  copySend(const TF16* src, TF16* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1, 0, 0, TF16, FUNCF16>(src, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, 0);
  }
  __device__ __forceinline__ void
  directCopySend(const TF16* src, TF16* dst, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1, 0, 0, TF16, FUNCF16>(src, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvCopySend(TF16* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1, 0, 0, TF16, FUNCF16>(NULL, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, 0);
  }

  __device__ __forceinline__ void
  recvCopySendAllGatherCompute(TF16* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1, 0, 1, TF16, FUNCF16>(NULL, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, 0);
  }

  __device__ __forceinline__ void
  directRecvCopySend(TF16* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1, 0, 0, TF16, FUNCF16>(NULL, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, directOffset);
  }

  __device__ __forceinline__ void
  directRecvCopySendAllGatherCompute(TF16* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1, 0, 1, TF16, FUNCF16>(NULL, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const TF16* src, TF32* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1, 0, 0, TF16, FUNCF16>(src, dst, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopy2(const TF16* src, TF16* dst, TF16 lr, TF16 beta1, TF16 beta2, TF16* m, TF16* v,  int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1, 1, 0, TF16, FUNCF16>(src, dst, lr, beta1, beta2, m, v,  nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceSend(const TF16* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0, 0, 0, TF16, FUNCF16>(src, NULL, (TF16)0, (TF16)0, (TF16)0, NULL, NULL,  nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const TF16* src, TF16* dst, TF32* w, TF32 lr, TF32 beta1, TF32 beta2, TF32* m, TF32* v, int nelem) {
    GenericOp2<0, 0, 1, 1, 1, 1, 1, 0>(src, dst, w, lr, beta1, beta2, m, v,  nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvReduceCopySend(const TF16* src, TF16* dst, TF16 lr, TF16 beta1, TF16 beta2, TF16* m, TF16* v,  ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1, 0, 0, TF16, FUNCF16>(src, dst, lr, beta1, beta2, m, v,  nelem, directOffset);
  }

__device__ __forceinline__ void
  recvReduceCopySendTF16(const TF16* src, TF16* dst, TF16 lr, TF16 beta1, TF16 beta2, TF16* m, TF16* v,  ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 0, 1, 1, 1, 1, 0, 0, TF16, FUNCF16>(src, dst, lr, beta1, beta2, m, v,  nelem, directOffset);
  }
  //TF32 
  __device__ __forceinline__ void
  sendF32(const TF32* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0, 0, 0, TF32, FUNCF32>(src, NULL, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, 0);
  }
  __device__ __forceinline__ void
  directSendF32(const TF32* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0, 0, 0, TF32, FUNCF32>(src, NULL, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, directOffset);
  }

  // __device__ __forceinline__ void
  // recvF32(TF32* dst, int nelem) {
  //   GenericOp2<0, 0, 1, 0, 0, 1, 0, 1>(NULL, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, 0);
  // }

    __device__ __forceinline__ void
  recvAllGatherComputeF32(TF32* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1, 0, 1, TF32, FUNCF32>(NULL, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, 0);
  }

  __device__ __forceinline__ void
  directRecvF32(TF32* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1, 0, 0, TF32, FUNCF32>(NULL, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, directOffset);
  }

  __device__ __forceinline__ void
  directRecvAllGatherComputeF32(TF32* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1, 0, 1, TF32, FUNCF32>(NULL, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, directOffset);
  }

  __device__ __forceinline__ void
  copySendF32(const TF32* src, TF32* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1, 0, 0, TF32, FUNCF32>(src, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, 0);
  }
  __device__ __forceinline__ void
  directCopySendF32(const TF32* src, TF32* dst, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1, 0, 0, TF32, FUNCF32>(src, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, directOffset);
  }

  // __device__ __forceinline__ void
  // recvCopySendF32(TF32* dst, int nelem) {
  //   GenericOp2<0, 0, 1, 1, 0, 1, 0, 1>(NULL, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, 0);
  // }

  __device__ __forceinline__ void
  recvCopySendAllGatherComputeF32(TF32* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1, 0, 1, TF32, FUNCF32>(NULL, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, 0);
  }

  __device__ __forceinline__ void
  directRecvCopySendF32(TF32* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1, 0, 0, TF32, FUNCF32>(NULL, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, directOffset);
  }

  __device__ __forceinline__ void
  directRecvCopySendAllGatherComputeF32(TF32* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1, 0, 1, TF32, FUNCF32>(NULL, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, directOffset);
  }

  __device__ __forceinline__ void
  recvReduceCopyF32(const TF32* src, TF32* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1, 0, 0, TF32, FUNCF32>(src, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopyF322(const TF32* src, TF32* dst, TF32 lr, TF32 beta1, TF32 beta2, TF32* m, TF32* v,  int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1, 1, 0, TF32, FUNCF32>(src, dst, lr, beta1, beta2, m, v,  nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceSendF32(const TF32* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0, 0, 0, TF32, FUNCF32>(src, NULL, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopySendF32(const TF32* src, TF32* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1, 0, 0, TF32, FUNCF32>(src, dst, (TF32)0, (TF32)0, (TF32)0, NULL, NULL,  nelem, 0);
  }
  __device__ __forceinline__ void
  directRecvReduceCopySendF32(const TF32* src, TF32* dst, TF32 lr, TF32 beta1, TF32 beta2, TF32* m, TF32* v,  ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1, 1, 0, TF32, FUNCF32>(src, dst, lr, beta1, beta2, m, v,  nelem, directOffset);
  }

  __device__ __forceinline__ ~ncclPrimitivesComputation() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};

#elif TYPE_PRIMS == 1


// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, class FUNC>
class ncclPrimitivesComputation {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  const int maxPartSize;
  const size_t totalSize;
  const size_t nChannels;
  const size_t loopSize;
  const int nranks;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;
  volatile uint64_t* recvConnTailPtr = NULL;
  uint64_t recvConnTail;
  uint64_t recvConnTailCache; // Cache last seen value

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  const T* recvDirectBuff[NRECV];
  T* sendDirectBuff[NSEND];
  const T* recvBuff[NRECV];
  T* sendBuff[NSEND];
  struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ const T* recvPtr(int i) { return ((const T*)recvBuff[i])+recvOffset(i); }
  inline __device__ T* sendPtr(int i) { return ((T*)sendBuff[i])+sendOffset(i); }

  inline __device__ const float* recvPtrFloat(int i) { return ((const float*)recvBuff[i])+recvOffset(i)/(sizeof(float)/sizeof(T)); }
  inline __device__ float* sendPtrFloat(int i) { return ((float*)sendBuff[i])+sendOffset(i)/(sizeof(float)/sizeof(T)); }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
  }
  inline __device__ void subBarrier() {
    asm volatile ("bar.sync 2, %0;" :: "r"(nthreads-WARP_SIZE));
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  inline __device__ void checkMismatch(struct ncclConnInfo* conn) {
    if (mismatch) {
      // In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch
      *(comm->fatalDevError) = ncclDevAssertedMismatch;
    } else if (conn && *conn->opCountRem > opCount) {
      mismatch += 1;
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort(int i, int send) {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *(comm->abortFlag);
      if (wid == i) checkMismatch(send ? sendConn : recvConn);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    spins = 0;
    mismatch = 0;
    if (sendConnHeadPtr) {
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + SLICESTEPS) {
        sendConnHeadCache = *sendConnHeadPtr;
        if (checkAbort(wid, 1)) break;
      }
      if (sendConnFifoPtr) {
        sendConnFifoPtr[sendConnHead%NCCL_STEPS] = nbytes;
      }
      sendConnHead += SLICESTEPS;
    }
  }

  inline __device__ void waitRecv() {
    spins = 0;
    mismatch = 0;
    if (recvConnTailPtr) {
      while (recvConnTailCache < recvConnTail + SLICESTEPS) {
        recvConnTailCache = *recvConnTailPtr;
        if (checkAbort(wid, 0)) break;
      }
      recvConnTail += SLICESTEPS;
    }
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += SLICESTEPS;
  }
  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += SLICESTEPS;
  }

  inline __device__ void incSend(int i) {
    sendStep[i] += SLICESTEPS;
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) *sendConnTailPtr = sendConnTail += SLICESTEPS;
  }

  template <int DIRECTRECV>
  inline __device__ const T* directRecvPtr(int i, ssize_t directOffset) {
    return DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : recvPtr(i);
  }

  template <int DIRECTSEND>
  inline __device__ T* directSendPtr(int i, ssize_t directOffset) {
    return DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : sendPtr(i);
  }

template <int DIRECTRECV>
  inline __device__ const float* directRecvPtrFloat(int i, ssize_t directOffset) {
    return DIRECTRECV && recvDirectBuff[i] ? (float*)recvDirectBuff[i]+directOffset : recvPtrFloat(i);
  }

  template <int DIRECTSEND>
  inline __device__ float* directSendPtrFloat(int i, ssize_t directOffset) {
    return DIRECTSEND && sendDirectBuff[i] ? (float*)sendDirectBuff[i]+directOffset : sendPtrFloat(i);
  }
  
  template <int DIRECTRECV>
  inline __device__ int directRecvInc(int i, int directInc, int sliceInc) {
    return DIRECTRECV && recvDirectBuff[i] ? directInc : sliceInc;
  }

  template <int DIRECTSEND>
  inline __device__ int directSendInc(int i, int directInc, int sliceInc) {
    return DIRECTSEND && sendDirectBuff[i] ? directInc : sliceInc;
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST, int COMPUTE, int ALLGATHER_COMPUTE>
  inline __device__ void
  GenericOpF32(const float* srcPtr, float* dstPtr, float lr, float beta1, float beta2, float* m, float* v, int nelem, ssize_t directOffset, int partNum, size_t totalSize, int dummy, float* rNorm, float* wNorm) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    const float* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? srcPtr : (float*)directRecvPtrFloat<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = (float*)recvPtrFloat(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = (float*)recvPtrFloat(i);
    }

    float* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? dstPtr : (float*)directSendPtrFloat<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = (float*)directSendPtrFloat<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = (float*)directSendPtrFloat<DIRECTSEND>(i, directOffset);
    }

    bool syncThread = tid >= nthreads-WARP_SIZE;
    const size_t slicedMomentSize = totalSize/nranks;
    const int numParts = (maxPartSize > 0) ? max(1UL, slicedMomentSize/maxPartSize) : 1;
    const int partsPerChannel = (totalSize < nranks * loopSize) ? 1 : DIVUP(totalSize, (nranks * loopSize));
    const size_t partIdx = blockIdx.x*partsPerChannel + partNum;

    //assert(numParts > 0);
    if (COMPUTE) {
      // if (threadIdx.x == 0) {
      //   printf("partsPerChannel %d partIdx %ld blockIdx.x %d nelem %d\n", partsPerChannel, partIdx, blockIdx.x, nelem);
      // }
      // if (partNum >= numParts) {
      //   printf("partNum %d numParts %d nelem %d\n", partNum, numParts, nelem);
      // }
      //assert (partNum < numParts);
    }
    const size_t partStartOffset = partIdx*maxPartSize;

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (!syncThread) {
        if (SEND) waitSend(realSize*sizeof(float));
        if (RECV) waitRecv();
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              ReduceOrCopyMulti<UNROLL, FuncSum<float>, float, 1, 1, 1, NSEND>(tid, nthreads-WARP_SIZE, 1, srcs, nsend, dsts+1, realSize);
            }
          } else {
            ReduceOrCopyMulti<UNROLL, FuncSum<float>, float, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST>(tid, nthreads-WARP_SIZE, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize);
          }
        }
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (syncThread) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
      dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST, int COMPUTE, int ALLGATHER_COMPUTE>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, T lr, T beta1, T beta2, T* m, T* v, int nelem, ssize_t directOffset, int partNum, size_t totalSize, int dummy, T* rNorm, T* wNorm) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    const T* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    bool syncThread = tid >= nthreads-WARP_SIZE;
    const size_t slicedMomentSize = totalSize/nranks;
    const int numParts = (maxPartSize > 0) ? max(1UL, slicedMomentSize/maxPartSize) : 1;
    const int partsPerChannel = (totalSize < nranks * loopSize) ? 1 : DIVUP(totalSize, (nranks * loopSize));
    const size_t partIdx = blockIdx.x*partsPerChannel + partNum;

    //assert(numParts > 0);
    if (COMPUTE) {
      // if (threadIdx.x == 0) {
      //   printf("partsPerChannel %d partIdx %ld blockIdx.x %d nelem %d\n", partsPerChannel, partIdx, blockIdx.x, nelem);
      // }
      // if (partNum >= numParts) {
      //   printf("partNum %d numParts %d nelem %d\n", partNum, numParts, nelem);
      // }
      //assert (partNum < numParts);
    }
    const size_t partStartOffset = partIdx*maxPartSize;

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (!syncThread) {
        if (SEND) waitSend(realSize*sizeof(T));
        if (RECV) waitRecv();
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              ReduceOrCopyMultiComputation<UNROLL, FUNC, T, 1, 1, 1, NSEND, COMPUTE, ALLGATHER_COMPUTE>(tid, nthreads-WARP_SIZE, 1, srcs, nsend, dsts+1, realSize, lr, beta1, beta2, m, v, directOffset+offset, partStartOffset, maxPartSize, rNorm, wNorm);
            }
          } else {
            ReduceOrCopyMultiComputation<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, COMPUTE, ALLGATHER_COMPUTE>(tid, nthreads-WARP_SIZE, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize, lr, beta1, beta2, m, v, directOffset+offset, partStartOffset, maxPartSize, rNorm, wNorm);
          }
        }
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (syncThread) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
      dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST, int COMPUTE, int ALLGATHER_COMPUTE>
  inline __device__ void
  GenericOp2(const T* gPtr, T* halfwPtr, float* wPtr, float lr, float beta1, float beta2, float* m, float* v, float* r, int nelem, ssize_t directOffset, int partNum, size_t totalSize, int dummy, float rNorm, float wNorm) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    const T* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? halfwPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? halfwPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    bool syncThread = tid >= nthreads-WARP_SIZE;
    const size_t slicedMomentSize = totalSize/nranks;
    const int numParts = (maxPartSize > 0) ? max(1UL, slicedMomentSize/maxPartSize) : 1;
    const int partsPerChannel = (totalSize < nranks * loopSize) ? 1 : DIVUP(totalSize, (nranks * loopSize));
    const size_t partIdx = blockIdx.x*partsPerChannel + partNum;

    //assert(numParts > 0);
    if (COMPUTE) {
      // if (threadIdx.x == 0) {
      //   printf("partsPerChannel %d partIdx %ld blockIdx.x %d nelem %d\n", partsPerChannel, partIdx, blockIdx.x, nelem);
      // }
      // if (partNum >= numParts) {
      //   printf("partNum %d numParts %d nelem %d\n", partNum, numParts, nelem);
      // }
      //assert (partNum < numParts);
    }
    const size_t partStartOffset = partIdx*maxPartSize;

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (!syncThread) {
        if (SEND) waitSend(realSize*sizeof(T));
        if (RECV) waitRecv();
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            // if (SEND) {
            //   ReduceOrCopyMultiComputation<UNROLL, FUNC, T, 1, 1, 1, NSEND, COMPUTE, ALLGATHER_COMPUTE>(tid, nthreads-WARP_SIZE, 1, srcs, nsend, dsts+1, realSize, lr, beta1, beta2, m, v, directOffset+offset, partStartOffset, maxPartSize, rNorm);
            // }
          } else {
            ReduceOrCopyMultiComputationForComputeSend<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, COMPUTE, ALLGATHER_COMPUTE>(tid, nthreads-WARP_SIZE, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize, lr, beta1, beta2, wPtr, m, v, r, directOffset+offset, partStartOffset, maxPartSize, rNorm, wNorm);
          }
        }
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (syncThread) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
      dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST, int COMPUTE, int ALLGATHER_COMPUTE>
  inline __device__ void
  GenericOpMixedPrecision(const T* gPtr, T* wPtr, float lr, float beta1, float beta2, float* weight, float* m, float* v, float* r, int nelem, ssize_t directOffset, int partNum, size_t totalSize, int dummy, float* rNorm, float* wNorm) {
    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);
    assert(COMPUTE == 1);
    const T* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? gPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? wPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    bool syncThread = tid >= nthreads-WARP_SIZE;
    const size_t slicedMomentSize = totalSize/nranks;
    const int numParts = (maxPartSize > 0) ? max(1UL, slicedMomentSize/maxPartSize) : 1;
    const int partsPerChannel = (totalSize < nranks * loopSize) ? 1 : DIVUP(totalSize, (nranks * loopSize));
    const size_t partIdx = blockIdx.x*partsPerChannel + partNum;

    //assert(numParts > 0);
    if (COMPUTE) {
      // if (threadIdx.x == 0) {
      //   printf("partsPerChannel %d partIdx %ld blockIdx.x %d nelem %d\n", partsPerChannel, partIdx, blockIdx.x, nelem);
      // }
      // if (partNum >= numParts) {
      //   printf("partNum %d numParts %d nelem %d\n", partNum, numParts, nelem);
      // }
      //assert (partNum < numParts);
    }
    const size_t partStartOffset = partIdx*maxPartSize;

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (!syncThread) {
        if (SEND) waitSend(realSize*sizeof(T));
        if (RECV) waitRecv();
        if (realSize > 0) {
          subBarrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            // if (SEND) {
            //   ReduceOrCopyMultiComputation<UNROLL, FUNC, T, 1, 1, 1, NSEND, COMPUTE, ALLGATHER_COMPUTE>(tid, nthreads-WARP_SIZE, 1, srcs, nsend, dsts+1, realSize, lr, beta1, beta2, m, v, directOffset+offset, partStartOffset, maxPartSize, rNorm);
            // }
          } else {
            ReduceOrCopyMultiComputationMixedPrecision<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, COMPUTE, ALLGATHER_COMPUTE>(tid, nthreads-WARP_SIZE, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, realSize, 
            lr, beta1, beta2, weight, m, v, r, directOffset+offset, partStartOffset, maxPartSize, rNorm, wNorm);
          }
        }
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (syncThread) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
      dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    recvBuff[i] = (const T*)conn->buff;
    recvStep[i] = conn->step;
    recvStep[i] = ROUNDUP(recvStep[i], SLICESPERCHUNK*SLICESTEPS);
    recvDirectBuff[i] = NULL;
    if (directBuff && (conn->direct & NCCL_DIRECT_GPU)) {
      recvDirectBuff[i] = directBuff;
      if (tid == 0) *conn->ptrExchange = directBuff;
    }
    if (wid == i) recvConn = conn;
    if (wid == i) recvConnTail = recvConnHead = recvStep[i]; // Make sure we set this after rounding up
    nrecv++;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= WARP_SIZE && tid < 2*WARP_SIZE && wid<nrecv) {
      recvConnTailPtr = recvConn->tail;
      recvConnTailCache = *recvConnTailPtr;
    }
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConnHeadPtr = recvConn->head;
      // Return credits in case we rounded up.
      *recvConnHeadPtr = recvConnHead;
      // Update opCount in case we skipped some operations
      *(recvConn->opCountLoc) = opCount;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    sendBuff[i] = (T*)conn->buff;
    sendStep[i] = conn->step;
    sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);
    sendDirectBuff[i] = NULL;
    if (directBuff && (conn->direct & NCCL_DIRECT_GPU)) {
      void* volatile* ptr = conn->ptrExchange;
      while ((sendDirectBuff[i] = (T*)(*ptr)) == NULL);
      barrier();
      if (tid == 0) *ptr = NULL;
    }
    if (wid == i) sendConn = conn;
    if (wid == i) sendConnTail = sendConnHead = sendStep[i]; // Make sure we set this after rounding up
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnFifoPtr = sendConn->fifo;
      *(sendConn->opCountLoc) = opCount;
    }
    if (tid >= nthreads-WARP_SIZE && wid<nsend) {
      sendConnTailPtr = sendConn->tail;
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConn->step = recvConnHead;
      *(recvConn->opCountLoc) = opCount+1;
      __threadfence_system();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      sendConn->step = sendConnHead;
      *(sendConn->opCountLoc) = opCount+1;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitivesComputation(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount, const size_t totalSize, const int nranks, int nChannels, size_t loopSize, int maxPartSize)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize), opCount(opCount), totalSize(totalSize), nranks(nranks), nChannels(nChannels), loopSize(loopSize), maxPartSize(maxPartSize) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, directBuff);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i, directBuff);
    loadRecvSync();
    loadSendSync();
  }

  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0, 0, 0>(src, NULL, (T)0, (T)0, (T)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }

  __device__ __forceinline__ void
  sendF32(const float* src, int nelem) {
    GenericOpF32<0, 0, 0, 1, 1, 0, 0, 0>(src, NULL, (float)0, (float)0, (float)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }

  __device__ __forceinline__ void
  computeSend(const T* g, T* halfw, float* w, float lr, float beta1, float beta2, float* m, float* v, float* r,  ssize_t directOffset, int nelem, int partNum, size_t size, int nChannels, float rNorm, float wNorm) {
    GenericOp2<0, 0, 0, 1, 1, 0, 1, 0>(g, halfw, w, lr, beta1, beta2, m, v, r, nelem, directOffset, partNum, size, 0, rNorm, wNorm);
  }

  __device__ __forceinline__ void
  directSend(const T* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0, 0, 0>(src, NULL, (T)0, (T)0, (T)0, NULL, NULL,  nelem, directOffset, 0, 0, 0, nullptr, nullptr);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1, 0, 0>(NULL, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }

    __device__ __forceinline__ void
  recvAllGatherCompute(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1, 0, 1>(NULL, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }

  __device__ __forceinline__ void
  directRecv(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1, 0, 0>(NULL, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, directOffset, 0, 0, 0, nullptr, nullptr);
  }

  __device__ __forceinline__ void
  directRecvAllGatherCompute(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1, 0, 1>(NULL, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, directOffset, 0, 0, 0, nullptr, nullptr);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1, 0, 0>(src, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }
  __device__ __forceinline__ void
  directCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1, 0, 0>(src, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, directOffset, 0, 0, 0, nullptr, nullptr);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1, 0, 0>(NULL, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }

  __device__ __forceinline__ void
  recvCopySendAllGatherCompute(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1, 0, 1>(NULL, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }

  __device__ __forceinline__ void
  directRecvCopySend(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1, 0, 0>(NULL, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, directOffset, 0, 0, 0, nullptr, nullptr);
  }

  __device__ __forceinline__ void
  directRecvCopySendAllGatherCompute(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1, 0, 1>(NULL, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, directOffset, 0, 0, 0, nullptr, nullptr);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1, 0, 0>(src, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }

  __device__ __forceinline__ void
  recvReduceCopyF32(const float* src, float* dst, int nelem) {
    GenericOpF32<0, 0, 1, 0, 1, 1, 0, 0>(src, dst, (float)0, (float)0, (float)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }

  __device__ __forceinline__ void
  recvReduceCopy2(const T* src, T* dst, T lr, T beta1, T beta2, T* m, T* v, size_t directOffset, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1, 1, 0>(src, dst, lr, beta1, beta2, m, v,  nelem, directOffset, 0, 0, 0, nullptr, nullptr);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0, 0, 0>(src, NULL, (T)0, (T)0, (T)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }

  __device__ __forceinline__ void
  recvReduceSendF32(const float* src, int nelem) {
    GenericOpF32<0, 0, 1, 1, 1, 0, 0, 0>(src, NULL, 0.0f, 0.0f, 0.0f, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1, 0, 0>(src, dst, (T)0, (T)0, (T)0, NULL, NULL,  nelem, 0, 0, 0, 0, nullptr, nullptr);;
  }
  __device__ __forceinline__ void
  directRecvReduceCopySend(const T* src, T* dst, float lr, float beta1, float beta2, float* weight, float* m, float* v, float* r, ssize_t directOffset, int nelem, int partNum, size_t size, int nChannels, float* rNorm, float* wNorm) {
    // Direct is only for the send part
    GenericOpMixedPrecision<0, 0, 1, 0, 1, 1, 1, 0>(src, dst, lr, beta1, beta2, weight, m, v, r, nelem, directOffset, partNum, size, 0, rNorm, wNorm);
  }

  __device__ __forceinline__ ~ncclPrimitivesComputation() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};

#endif 

#include "prims_ll_computation.h"
//#include "prims_ll128.h"

#endif
