/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "op128.h"

#define TYPE_PRIMS_LL128 1











#if TYPE_PRIMS_LL128 == 0

  #define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS-1)

  template <typename T, class FUNC, int NRECV, int NSEND>
  class ncclLL128PrimitivesComputation {
  private:
    const int tid;
    const int nthreads;
    const int wid;
    const int warp;
    const size_t totalSize;
    const bool flagThread;
    int nrecv = 0;
    int nsend = 0;
    struct ncclConnInfo* recvConn = NULL;
    volatile uint64_t* recvConnHeadPtr = NULL;
    uint64_t recvConnHead;

    struct ncclConnInfo* sendConn = NULL;
    volatile int* sendConnFifoPtr = NULL;
    volatile uint64_t* sendConnTailPtr = NULL;
    uint64_t sendConnTail;
    volatile uint64_t* sendConnHeadPtr = NULL;
    uint64_t sendConnHead;
    uint64_t sendConnHeadCache; // Cache last seen value

    uint64_t recvStep[NRECV];
    uint64_t sendStep[NSEND];
    uint64_t* recvBuff[NRECV];
    uint64_t* sendBuff[NSEND];
    struct ncclDevComm* comm;

    volatile uint64_t* shmem;

    inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*NCCL_LL128_SLICE_ELEMS; }
    inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*NCCL_LL128_SLICE_ELEMS; }
    inline __device__ uint64_t* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
    inline __device__ uint64_t* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
    inline __device__ uint64_t recvFlag(int i) { return recvStep[i]+1; }
    inline __device__ uint64_t sendFlag(int i) { return sendStep[i]+1; }

    inline __device__ void barrier() {
      if (NSEND>NRECV) {
        asm volatile ("bar.sync 2, %0;" :: "r"(nthreads));
      } else {
        asm volatile ("bar.sync 3, %0;" :: "r"(nthreads));
      }
    }

    uint32_t mismatch = 0;
    const uint64_t opCount;

    inline __device__ void checkMismatch(struct ncclConnInfo* conn) {
      if (mismatch > 20) {
        // We have seen that the peer advanced opcount so many times yet we are still waiting for credit of current op, so it is _most likely_ a mismatch
        // Note that we are not using _threadfence_system in LL so the error cannot be asserted
        *(comm->fatalDevError) = ncclDevSuspectedMismatch;
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
        while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
          sendConnHeadCache = *sendConnHeadPtr;
          if (checkAbort(wid, 1)) break;
        }
        if (sendConnFifoPtr) {
          sendConnFifoPtr[sendStep[wid]%NCCL_STEPS] = nbytes;
        }
        sendConnHead += 1;
      }
    }

    inline __device__ void incRecv(int i) {
      recvStep[i] += 1;
    }
    inline __device__ void postRecv() {
      if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
    }

    inline __device__ void incSend(int i) {
      sendStep[i] += 1;
    }
    inline __device__ void postSend() {
      if (sendConnTailPtr) { __threadfence(); *sendConnTailPtr = sendConnTail += 1; }
    }

    template <int ELEMS_PER_THREAD>
    inline __device__ void loadSrcToShmem128(int maxOffset, const uint64_t* src64Ptr) {
  #if 0
      uint64_t v[ELEMS_PER_THREAD];
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        if (u*WARP_SIZE < maxOffset) load128(src64Ptr+u*WARP_SIZE, v[u], v[u+1]);
      }
      uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        storeShmem128(shmemAsmPtr+u*WARP_SIZE, v[u], v[u+1]);
      }
  #else
      uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        if (u*WARP_SIZE < maxOffset) {
          uint64_t v0, v1;
          load128(src64Ptr+u*WARP_SIZE, v0, v1);
          storeShmem128(shmemAsmPtr+u*WARP_SIZE, v0, v1);
        }
      }
  #endif
    }

    inline __device__ void loadSrcToShmem(int start, int end, const T* srcPtr) {
      T* shmemPtr = (T*)(shmem-2*wid);
      for (int offset = start+wid; offset < end; offset += WARP_SIZE) {
        shmemPtr[offset] = srcPtr[offset];
      }
    }

    template <int ELEMS_PER_THREAD>
    inline __device__ void storeShmemToDst128(int maxOffset, uint64_t* dst64Ptr) {
      uint64_t v[ELEMS_PER_THREAD];
      uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        loadShmem128(shmemAsmPtr+u*WARP_SIZE, v[u], v[u+1]);
      }
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        if (u*WARP_SIZE < maxOffset) store128(dst64Ptr+u*WARP_SIZE, v[u], v[u+1]);
      }
    }

    inline __device__ void storeShmemToDst(int start, int end, T* dstPtr) {
      T* shmemPtr = (T*)(shmem-2*wid);
      for (int offset = start+wid; offset < end; offset += WARP_SIZE) {
        dstPtr[offset] = shmemPtr[offset];
      }
    }

    #define WARP_MASK 0xffffffff

    template <int ELEMS_PER_THREAD, int RECV, int SEND, int SRC, int DST, int COMPUTE, int ALLGATHER_COMPUTE>
    __device__ __forceinline__ void recvReduceSendCopy2(float lr, float beta1, float beta2, uint64_t* gPack, uint64_t* halfWPack, 
                                                        Pack128* wPack, Pack128* mPack, Pack128* vPack, size_t sumOffset, int ll128Offset) {
      uint64_t v[ELEMS_PER_THREAD];

      /************* Data Loading : SHMEM -> REG **************/
      if (SRC) {
        volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u] = shmem64Ptr[u*(WARP_SIZE-2)];
          if (!flagThread) v[u+1] = shmem64Ptr[u*(WARP_SIZE-2)+1];
        }
      }
      /*********** End Data Loading : SHMEM -> REG ************/

      /************************ Recv **************************/
      if (RECV) {
        uint64_t flag = recvFlag(0);
        uint64_t* ptr = recvPtr(0)+ll128Offset;
        bool needReload;
        uint64_t v0, v1;
        do {
          needReload = false;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, v0, v1);
            needReload |= flagThread && (v1 != flag);
          }
        } while (__any_sync(WARP_MASK, needReload) && checkAbort(0, 0) == 0);
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, v0, v1);
          v[u] = SRC ? MULTI<FUNC, T>()(v0, v[u]) : v0;
          v[u+1] = SRC ? MULTI<FUNC, T>()(v1, v[u+1]) : v1;
        }

        for (int i=1; i<NRECV && i<nrecv; i++) {
          uint64_t flag = recvFlag(i);
          uint64_t* ptr = recvPtr(i)+ll128Offset;
          uint64_t v0, v1;
          do {
            needReload = false;
            #pragma unroll
            for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
              load128(ptr+u*WARP_SIZE, v0, v1);
              needReload |= flagThread && (v1 != flag);
            }
          } while (__any_sync(WARP_MASK, needReload) && checkAbort(i, 0) == 0);
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, v0, v1);
            v[u] = MULTI<FUNC, T>()(v0, v[u]);
            v[u+1] = MULTI<FUNC, T>()(v1, v[u+1]);
          }
        }
      }
      /********************** End Recv ************************/


      /************* Data Storing : REG -> SHMEM **************/
      if (DST) {
        volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
        
        // static_assert((COMPUTE ^ ALLGATHER_COMPUTE) || (!COMPUTE && !ALLGATHER_COMPUTE));
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          if (COMPUTE) {
            int wOffset = u*(WARP_SIZE-2) - (2*wid)/NCCL_LL128_LINEELEMS;
            ssize_t totalOffset = sumOffset + wOffset*(ssize_t)(sizeof(Pack128)/sizeof(float));
            if (totalOffset < totalSize) {
              Pack128 readVal;
              Fetch128(readVal, (wPack+wOffset));
              Pack128 vval;
              Fetch128(vval, (vPack+wOffset));
              Pack128 mval;
              Fetch128(mval, (mPack+wOffset));

              Pack128 S5;
              uint2 __val = *(reinterpret_cast<const uint2*>(&v[u]));
              Pack128 S4, S3, S2, S1;

              MULTI128<mixedbinOp1<float>, half>().mixedbinOp1(beta2, vval, __val, S2);
              MULTI128<binOp2<float>, half>().binOp2(beta2, S2, S4);

              MULTI128<mixedbinOp3<float>, half>().mixedbinOp1(beta2, mval, __val, S1);
              MULTI128<binOp2<float>, half>().binOp4(beta1, S1, S3);
              
              MULTI128<delta<float>, half>().delta(lr, S3, S4, S5);

              Store128(vPack+wOffset, S2);
              Store128(mPack+wOffset, S1);

              Pack128 finalVal;
              MULTI128<weightUpdate<float>, half>().weightUpdate(readVal, S5, finalVal);
              Store128(wPack+wOffset, finalVal);
              v[u] = float4ToHalf4(finalVal);
            }

            if (!flagThread) {
              int wOffset = u*(WARP_SIZE-2) + 1 - (2*wid)/NCCL_LL128_LINEELEMS;
              ssize_t totalOffset = sumOffset + wOffset*(ssize_t)(sizeof(Pack128)/sizeof(float));
              if (totalOffset < totalSize) {
                Pack128 readVal;
                Fetch128(readVal, (wPack+wOffset));
                Pack128 vval;
                Fetch128(vval, (vPack+wOffset));
                Pack128 mval;
                Fetch128(mval, (mPack+wOffset));

                Pack128 S5;
                uint2 __val = *(reinterpret_cast<const uint2*>(&v[u+1]));
                Pack128 S4, S3, S2, S1;

                MULTI128<mixedbinOp1<float>, half>().mixedbinOp1(beta2, vval, __val, S2);
                MULTI128<binOp2<float>, half>().binOp2(beta2, S2, S4);

                MULTI128<mixedbinOp3<float>, half>().mixedbinOp1(beta2, mval, __val, S1);
                MULTI128<binOp2<float>, half>().binOp4(beta1, S1, S3);
                
                MULTI128<delta<float>, half>().delta(lr, S3, S4, S5);

                Store128(vPack+wOffset, S2);
                Store128(mPack+wOffset, S1);

                Pack128 finalVal;
                MULTI128<weightUpdate<float>, half>().weightUpdate(readVal, S5, finalVal);
                Store128(wPack+wOffset, finalVal);
                v[u+1] = float4ToHalf4(finalVal);
              }
            }
          }
          
          shmem64Ptr[u*(WARP_SIZE-2)] = v[u];
          if (!flagThread) shmem64Ptr[u*(WARP_SIZE-2)+1] = v[u+1];
        }
      }
      /*********** End data Storing : REG -> SHMEM ************/

      
      /************************ Send **************************/
      if (SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) {
          int flag = sendFlag(i);
          uint64_t* ptr = sendPtr(i)+ll128Offset;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
          }
        }
        int flag = sendFlag(0);
        uint64_t* ptr = sendPtr(0)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
        }
      }
      /********************** End Send ************************/
    }


    template <int ELEMS_PER_THREAD, int RECV, int SEND, int SRC, int DST, int COMPUTE>
    __device__ __forceinline__ void recvReduceSendCopy(int ll128Offset) {
      uint64_t v[ELEMS_PER_THREAD];

      /************* Data Loading : SHMEM -> REG **************/
      if (SRC) {
        volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u] = shmem64Ptr[u*(WARP_SIZE-2)];
          if (!flagThread) v[u+1] = shmem64Ptr[u*(WARP_SIZE-2)+1];
        }
      }
      /*********** End Data Loading : SHMEM -> REG ************/

      /************************ Recv **************************/
      if (RECV) {
        uint64_t flag = recvFlag(0);
        uint64_t* ptr = recvPtr(0)+ll128Offset;
        bool needReload;
        uint64_t v0, v1;
        do {
          needReload = false;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, v0, v1);
            needReload |= flagThread && (v1 != flag);
          }
        } while (__any_sync(WARP_MASK, needReload) && checkAbort(0, 0) == 0);
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, v0, v1);
          v[u] = SRC ? MULTI<FUNC, T>()(v0, v[u]) : v0;
          v[u+1] = SRC ? MULTI<FUNC, T>()(v1, v[u+1]) : v1;
        }

        for (int i=1; i<NRECV && i<nrecv; i++) {
          uint64_t flag = recvFlag(i);
          uint64_t* ptr = recvPtr(i)+ll128Offset;
          uint64_t v0, v1;
          do {
            needReload = false;
            #pragma unroll
            for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
              load128(ptr+u*WARP_SIZE, v0, v1);
              needReload |= flagThread && (v1 != flag);
            }
          } while (__any_sync(WARP_MASK, needReload) && checkAbort(i, 0) == 0);
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, v0, v1);
            v[u] = MULTI<FUNC, T>()(v0, v[u]);
            v[u+1] = MULTI<FUNC, T>()(v1, v[u+1]);
          }
        }
      }
      /********************** End Recv ************************/

      /************************ Send **************************/
      if (SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) {
          int flag = sendFlag(i);
          uint64_t* ptr = sendPtr(i)+ll128Offset;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
          }
        }
        int flag = sendFlag(0);
        uint64_t* ptr = sendPtr(0)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
        }
      }
      /********************** End Send ************************/

      /************* Data Storing : REG -> SHMEM **************/
      if (DST) {
        volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          shmem64Ptr[u*(WARP_SIZE-2)] = v[u];
          if (!flagThread) shmem64Ptr[u*(WARP_SIZE-2)+1] = v[u+1];
        }
      }
      /*********** End data Storing : REG -> SHMEM ************/
    }

    #define LL128INC (WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD)
    #define ELEMINC (LL128INC-(LL128INC/NCCL_LL128_LINEELEMS))


    template <int RECV, int SEND, int SRC, int DST, int COMPUTE, int ALLGATHER_COMPUTE>
    __device__ void GenericOp2(float lr, float beta1, float beta2, const T* g, T* halfw, float* w, float* m, float* v, int startOffset, int nelem) {
      if (nelem <= 0) {
        // Don't move any data but still increase steps and sync with prev/next
        if (SEND) waitSend(0);
        FOR_SEND(incSend); if (SEND) postSend();
        FOR_RECV(incRecv); if (RECV) postRecv();
        return;
      }
      const int nelem64 = ((nelem*sizeof(T))/(2*sizeof(uint64_t)))*2;
      uint64_t* gPack = (uint64_t*)g;
      uint64_t* halfwPack = (uint64_t*)halfw;
      Pack128* wPack = (Pack128*)w;
      Pack128* mPack = (Pack128*)m;
      Pack128* vPack = (Pack128*)v;


      int ll128Offset = LL128INC*warp+2*wid;
      int elemOffset = ELEMINC*warp;
      const int nwarps = nthreads/WARP_SIZE;

      if (SEND) waitSend(DIVUP(nelem*sizeof(T), ELEMINC*sizeof(uint64_t))*LL128INC*sizeof(uint64_t));
      barrier();

      while (elemOffset*(sizeof(uint64_t)/sizeof(T)) < nelem) {
        const int maxOffset128 = min(nelem64-elemOffset, (int)ELEMINC);
        const int maxOffset = min(nelem-(elemOffset*((int)(sizeof(uint64_t)/sizeof(T)))), (int)(ELEMINC*(sizeof(uint64_t)/sizeof(T))));
        if (SRC) {
          int done = 0;
          if ((((uint64_t)g)&0xf) == 0) {
            loadSrcToShmem128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, gPack+elemOffset+2*wid);
            done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
          }
          loadSrcToShmem(done, maxOffset, (T*)(gPack+elemOffset));
        }
        __syncwarp();
        recvReduceSendCopy2<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SRC, DST, COMPUTE, ALLGATHER_COMPUTE>(lr, beta1, beta2, gPack, halfwPack+elemOffset+2*wid, wPack+elemOffset+2*wid, mPack+elemOffset+2*wid, vPack+elemOffset+2*wid, startOffset+(elemOffset+2*wid)*sizeof(Pack128)/sizeof(float), ll128Offset);
        __syncwarp();
        if (DST) {
          int done = 0;
          if ((((uint64_t)w)&0xf) == 0) {
            storeShmemToDst128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, halfwPack+elemOffset+2*wid);
            done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
          }
          storeShmemToDst(done, maxOffset, (T*)(halfwPack+elemOffset));
        }
        __syncwarp();
        ll128Offset += LL128INC*nwarps;
        elemOffset += ELEMINC*nwarps;
      }

      barrier();
      FOR_SEND(incSend); if (SEND) postSend();
      FOR_RECV(incRecv); if (RECV) postRecv();
    }

    template <int RECV, int SEND, int SRC, int DST>
    __device__ void GenericOp(const T* srcPtr, T* dstPtr, int nelem) {
      if (nelem <= 0) {
        // Don't move any data but still increase steps and sync with prev/next
        if (SEND) waitSend(0);
        FOR_SEND(incSend); if (SEND) postSend();
        FOR_RECV(incRecv); if (RECV) postRecv();
        return;
      }
      const int nelem64 = ((nelem*sizeof(T))/(2*sizeof(uint64_t)))*2;
      const uint64_t* src64Ptr = ((uint64_t*)srcPtr);
      uint64_t* dst64Ptr = ((uint64_t*)dstPtr);

      int ll128Offset = LL128INC*warp+2*wid;
      int elemOffset = ELEMINC*warp;
      const int nwarps = nthreads/WARP_SIZE;

      if (SEND) waitSend(DIVUP(nelem*sizeof(T), ELEMINC*sizeof(uint64_t))*LL128INC*sizeof(uint64_t));
      barrier();

      while (elemOffset*(sizeof(uint64_t)/sizeof(T)) < nelem) {
        const int maxOffset128 = min(nelem64-elemOffset, (int)ELEMINC);
        const int maxOffset = min(nelem-(elemOffset*((int)(sizeof(uint64_t)/sizeof(T)))), (int)(ELEMINC*(sizeof(uint64_t)/sizeof(T))));
        if (SRC) {
          int done = 0;
          if ((((uint64_t)srcPtr)&0xf) == 0) {
            loadSrcToShmem128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, src64Ptr+elemOffset+2*wid);
            done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
          }
          loadSrcToShmem(done, maxOffset, (T*)(src64Ptr+elemOffset));
        }
        __syncwarp();
        recvReduceSendCopy<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SRC, DST, 0>(ll128Offset);
        __syncwarp();
        if (DST) {
          int done = 0;
          if ((((uint64_t)dstPtr)&0xf) == 0) {
            storeShmemToDst128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, dst64Ptr+elemOffset+2*wid);
            done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
          }
          storeShmemToDst(done, maxOffset, (T*)(dst64Ptr+elemOffset));
        }
        __syncwarp();
        ll128Offset += LL128INC*nwarps;
        elemOffset += ELEMINC*nwarps;
      }

      barrier();
      FOR_SEND(incSend); if (SEND) postSend();
      FOR_RECV(incRecv); if (RECV) postRecv();
    }

    __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
      recvBuff[i] = conn->ll128Buff;
      recvStep[i] = conn->step;
      if (wid == i) recvConn = conn;
      nrecv++;
    }
    __device__ __forceinline__ void loadRecvSync() {
      if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
        recvConnHeadPtr = recvConn->head;
        recvConnHead = recvConn->step;
        // Update opCount in case we skipped some operations
        *(recvConn->opCountLoc) = opCount;
      }
    }

    __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
      sendBuff[i] = conn->ll128Buff;
      sendStep[i] = conn->step;
      if (wid == i) sendConn = conn;
      nsend++;
    }
    __device__ __forceinline__ void loadSendSync() {
      if (tid < nsend) {
        sendConnHeadPtr = sendConn->head;
        sendConnHeadCache = *sendConnHeadPtr;
        sendConnHead = sendConn->step;
        sendConnFifoPtr = sendConn->fifo;
        *(sendConn->opCountLoc) = opCount;
      }
      if (tid >= nthreads-WARP_SIZE && wid<nsend) {
        if (sendConn->fifo) {
          sendConnTailPtr = sendConn->tail;
          sendConnTail = sendConn->step;
        }
      }
    }

    __device__ __forceinline__ void saveRecvSync() {
      if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
        recvConn->step = recvConnHead;
        *(recvConn->opCountLoc) = opCount+1;
        __threadfence_block();
      }
    }

    __device__ __forceinline__ void saveSendSync() {
      if (tid < nsend) {
        sendConn->step = sendConnHead;
        *(sendConn->opCountLoc) = opCount+1;
        __threadfence_block();
      }
    }

  public:
    __device__ __forceinline__
    ncclLL128PrimitivesComputation(const int tid, const int nthreads, int* recvPeers, int* sendPeers, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount, const size_t totalSize)
      : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), warp(tid/WARP_SIZE), flagThread((tid%8)==7), opCount(opCount), shmem(ncclShmem+(threadIdx.x/WARP_SIZE)*NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE+2*wid), totalSize(totalSize) {
      // Make sure step is updated before we read it.
      barrier();

      for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);
      for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
      loadRecvSync();
      loadSendSync();
    }

    __device__ void send(const T* src, int nelem) {
      return GenericOp<0, 1, 1, 0>(src, NULL, nelem);
    }

    __device__ void recv(T* dst, int nelem) {
      return GenericOp<1, 0, 0, 1>(NULL, dst, nelem);
    }

    __device__ void recvAllGatherCompute(/*PRIMSLL128: {INSERT recv ARGS}*/ int nelem) {
      return GenericOp2<1, 0, 0, 1, 0, 1>(/*{INSERT GenericOp2 CALL PARAMS for ALLGATHER COMPUTE}*/ nelem);
    }

    __device__ void recvReduceSend(const T* src, int nelem) {
      return GenericOp<1, 1, 1, 0>(src, NULL, nelem);
    }

    __device__ void recvReduceCopy(const T* src, T* dst, int nelem) {
      return GenericOp<1, 0, 1, 1>(src, dst, nelem);
    }

    __device__ void recvReduceCopy2(T lr, T beta1, T beta2, const T* g, T* w, T* m, T* v,  int nelem) {
      return GenericOp2<1, 0, 1, 1, 1, 0>(lr, beta1, beta2, g, w, m, v,  nelem);
    }

    __device__ void recvReduceCopyAllGatherCompute(/*PRIMSLL128: {INSERT recv ARGS}*/ int nelem) {
      return GenericOp2<1, 0, 1, 1, 0, 1>(/*{INSERT GenericOp2 CALL PARAMS for ALLGATHER COMPUTE}*/ nelem);
    }

    __device__ void copySend(const T* src, T* dst, int nelem) {
      return GenericOp<0, 1, 1, 1>(src, dst, nelem);
    }

    __device__ void recvCopySend(T* dst, int nelem) {
      return GenericOp<1, 1, 0, 1>(NULL, dst, nelem);
    }

    __device__ void recvCopySendAllGatherCompute(/*PRIMSLL128: {INSERT recv ARGS}*/ int nelem) {
      return GenericOp2<1, 1, 0, 1, 0, 1>(/*{INSERT GenericOp2 CALL PARAMS for ALLGATHER COMPUTE}*/nelem);
    }


    __device__ void recvReduceCopySend(float lr, float beta1, float beta2, const T* g, T* halfw, float* w, float* m, float* v, int offset, int nelem) {
      return GenericOp2<1, 1, 1, 1, 1, 0>(lr, beta1, beta2, g, halfw, w, m, v, offset, nelem);
    }

    __device__ __forceinline__ ~ncclLL128PrimitivesComputation() {
      // Save steps for the next operation
      saveRecvSync();
      saveSendSync();
    }
  };
#elif TYPE_PRIMS_LL128 == 1
  /*************************************************************************
  * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
  *
  * See LICENSE.txt for license information
  ************************************************************************/

  #include "op128.h"

  #define NCCL_LL128_FLAGTHREAD (NCCL_LL128_LINEELEMS-1)

  template <typename T, class FUNC, int NRECV, int NSEND>
  class ncclLL128PrimitivesComputation {
  private:
    const size_t totalSize;
    const int tid;
    const int nthreads;
    const int wid;
    const int warp;
    const bool flagThread;
    const int nranks;
    const int nChannels;
    const ssize_t loopSize;
    const ssize_t maxPartSize;
    int nrecv = 0;
    int nsend = 0;
    struct ncclConnInfo* recvConn = NULL;
    volatile uint64_t* recvConnHeadPtr = NULL;
    uint64_t recvConnHead;

    struct ncclConnInfo* sendConn = NULL;
    volatile int* sendConnFifoPtr = NULL;
    volatile uint64_t* sendConnTailPtr = NULL;
    uint64_t sendConnTail;
    volatile uint64_t* sendConnHeadPtr = NULL;
    uint64_t sendConnHead;
    uint64_t sendConnHeadCache; // Cache last seen value

    uint64_t recvStep[NRECV];
    uint64_t sendStep[NSEND];
    uint64_t* recvBuff[NRECV];
    uint64_t* sendBuff[NSEND];
    struct ncclDevComm* comm;

    volatile uint64_t* shmem;

    inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*NCCL_LL128_SLICE_ELEMS; }
    inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*NCCL_LL128_SLICE_ELEMS; }
    inline __device__ uint64_t* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
    inline __device__ uint64_t* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
    inline __device__ uint64_t recvFlag(int i) { return recvStep[i]+1; }
    inline __device__ uint64_t sendFlag(int i) { return sendStep[i]+1; }

    inline __device__ void barrier() {
      if (NSEND>NRECV) {
        asm volatile ("bar.sync 2, %0;" :: "r"(nthreads));
      } else {
        asm volatile ("bar.sync 3, %0;" :: "r"(nthreads));
      }
    }

    uint32_t mismatch = 0;
    const uint64_t opCount;

    inline __device__ void checkMismatch(struct ncclConnInfo* conn) {
      if (mismatch > 20) {
        // We have seen that the peer advanced opcount so many times yet we are still waiting for credit of current op, so it is _most likely_ a mismatch
        // Note that we are not using _threadfence_system in LL so the error cannot be asserted
        *(comm->fatalDevError) = ncclDevSuspectedMismatch;
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
        while (sendConnHeadCache + NCCL_STEPS < sendConnHead + 1) {
          sendConnHeadCache = *sendConnHeadPtr;
          if (checkAbort(wid, 1)) break;
        }
        if (sendConnFifoPtr) {
          sendConnFifoPtr[sendStep[wid]%NCCL_STEPS] = nbytes;
        }
        sendConnHead += 1;
      }
    }

    inline __device__ void incRecv(int i) {
      recvStep[i] += 1;
    }
    inline __device__ void postRecv() {
      if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
    }

    inline __device__ void incSend(int i) {
      sendStep[i] += 1;
    }
    inline __device__ void postSend() {
      if (sendConnTailPtr) { __threadfence(); *sendConnTailPtr = sendConnTail += 1; }
    }

    template <int ELEMS_PER_THREAD>
    inline __device__ void loadSrcToShmem128(int maxOffset, const uint64_t* src64Ptr) {
  #if 0
      uint64_t v[ELEMS_PER_THREAD];
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        if (u*WARP_SIZE < maxOffset) load128(src64Ptr+u*WARP_SIZE, v[u], v[u+1]);
      }
      uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        storeShmem128(shmemAsmPtr+u*WARP_SIZE, v[u], v[u+1]);
      }
  #else
      uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        if (u*WARP_SIZE < maxOffset) {
          uint64_t v0, v1;
          load128(src64Ptr+u*WARP_SIZE, v0, v1);
          storeShmem128(shmemAsmPtr+u*WARP_SIZE, v0, v1);
        }
      }
  #endif
    }

    template<typename T1>
    inline __device__ void loadSrcToShmem(int start, int end, const T1* srcPtr) {
      T1* shmemPtr = (T1*)(shmem-2*wid);
      for (int offset = start+wid; offset < end; offset += WARP_SIZE) {
        shmemPtr[offset] = srcPtr[offset];
      }
    }

    template <int ELEMS_PER_THREAD>
    inline __device__ void storeShmemToDst128(int maxOffset, uint64_t* dst64Ptr) {
      uint64_t v[ELEMS_PER_THREAD];
      uint64_t* shmemAsmPtr = shmemCvtPtr(shmem);
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        loadShmem128(shmemAsmPtr+u*WARP_SIZE, v[u], v[u+1]);
      }
      #pragma unroll
      for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
        if (u*WARP_SIZE < maxOffset) store128(dst64Ptr+u*WARP_SIZE, v[u], v[u+1]);
      }
    }

    template<typename T1>
    inline __device__ void storeShmemToDst(int start, int end, T1* dstPtr) {
      T1* shmemPtr = (T1*)(shmem-2*wid);
      for (int offset = start+wid; offset < end; offset += WARP_SIZE) {
        dstPtr[offset] = shmemPtr[offset];
      }
    }

    #define WARP_MASK 0xffffffff

    union converter {
      uint64_t storage;
      struct pair {
        float x, y;
      } floats;
    };
    
    inline __device__ void uint64_to_floats(uint64_t d, float* x, float* y) {
      converter c;
      c.storage = d;
      *x = c.floats.x;
      *y = c.floats.y;
    }

    template <int ELEMS_PER_THREAD, int RECV, int SEND, int SRC, int DST, int COMPUTE, int LAMB_SEND_COMPUTE>
    __device__ __forceinline__ void recvReduceSendCopy2(float lr, float beta1, float beta2, uint64_t* gPack, uint64_t* halfWPack, 
                                                        Pack128* wPack, Pack128* mPack, Pack128* vPack, Pack128* rPack, size_t sumOffset, 
                                                        int ll128Offset, float *wVal, float* rVal, float scale)  {
      uint64_t v[ELEMS_PER_THREAD];

      /************* Data Loading : SHMEM -> REG **************/
      if (SRC) {
        volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u] = shmem64Ptr[u*(WARP_SIZE-2)];
          if (!flagThread) v[u+1] = shmem64Ptr[u*(WARP_SIZE-2)+1];
        }
      }
      /*********** End Data Loading : SHMEM -> REG ************/

      /************************ Recv **************************/
      if (RECV) {
        uint64_t flag = recvFlag(0);
        uint64_t* ptr = recvPtr(0)+ll128Offset;
        bool needReload;
        uint64_t v0, v1;
        do {
          needReload = false;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, v0, v1);
            needReload |= flagThread && (v1 != flag);
          }
        } while (__any_sync(WARP_MASK, needReload) && checkAbort(0, 0) == 0);
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, v0, v1);
          v[u] = SRC ? MULTI<FUNC, T>()(v0, v[u]) : v0;
          v[u+1] = SRC ? MULTI<FUNC, T>()(v1, v[u+1]) : v1;
        }

        for (int i=1; i<NRECV && i<nrecv; i++) {
          uint64_t flag = recvFlag(i);
          uint64_t* ptr = recvPtr(i)+ll128Offset;
          uint64_t v0, v1;
          do {
            needReload = false;
            #pragma unroll
            for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
              load128(ptr+u*WARP_SIZE, v0, v1);
              needReload |= flagThread && (v1 != flag);
            }
          } while (__any_sync(WARP_MASK, needReload) && checkAbort(i, 0) == 0);
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, v0, v1);
            v[u] = MULTI<FUNC, T>()(v0, v[u]);
            v[u+1] = MULTI<FUNC, T>()(v1, v[u+1]);
          }
        }
      }
      /********************** End Recv ************************/


      /************* Data Storing : REG -> SHMEM **************/
      if (DST || COMPUTE || LAMB_SEND_COMPUTE) {
        volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          if (COMPUTE || LAMB_SEND_COMPUTE) {
            int wOffset = u*(WARP_SIZE-2) - (2*wid)/NCCL_LL128_LINEELEMS;

            size_t totalOffset = sumOffset + (wOffset)*(sizeof(Pack128)/sizeof(float));
            // const size_t slicedMomentSize = totalSize/nranks;
            // const int numParts = (maxPartSize > 0) ? max(1UL, slicedMomentSize/maxPartSize) : 1;
            // const int partsPerChannel = (totalSize < loopSize) ? 1 : DIVUP(totalSize, loopSize);
            // const size_t partIdx = blockIdx.x*partsPerChannel + partNum;

            // const size_t partStartOffset = partIdx*maxPartSize;

            assert(maxPartSize > 0);
            
            if (totalOffset < totalSize) {
              if (COMPUTE) {
                Pack128 wght;
                Fetch128(wght, (wPack+wOffset));
                Pack128 vval;
                Fetch128(vval, (vPack+wOffset));
                Pack128 mval;
                Fetch128(mval, (mPack+wOffset));

                Pack128 S5;
                uint2 __val = *(reinterpret_cast<const uint2*>(&v[u]));
                Pack128 S4, S3, S2, S1;

                MULTI128<mixedbinOp1<float>, half>().mixedbinOp1(beta2, vval, __val, S2);
                MULTI128<binOp2<float>, float>().binOp2(beta2, S2, S4);

                MULTI128<mixedbinOp3<float>, half>().mixedbinOp1(beta2, mval, __val, S1);
                MULTI128<binOp2<float>, float>().binOp4(beta1, S1, S3);
                
                Store128(vPack+wOffset, S2);
                Store128(mPack+wOffset, S1);

                Pack128 r;
                MULTI128<rOp<float>, float>().r(wght, S3, S4, r);
                Store128(rPack+wOffset, r);

                float4 f4 = *(reinterpret_cast<float4*>(&wght));
                *wVal += f4.x*f4.x + f4.y*f4.y + f4.z*f4.z + f4.w*f4.w;

                f4 = *(reinterpret_cast<float4*>(&r));
                *rVal += f4.x*f4.x + f4.y*f4.y + f4.x*f4.x + f4.y*f4.y;
                
              } else {
                Pack128 rLambdaW;
                Fetch128(rLambdaW, rPack+wOffset);

                Pack128 weightVal;
                Fetch128(weightVal, wPack+wOffset);

                Pack128 finalVal;
                MULTI128<delta<float>, float>().delta(lr*scale, rLambdaW, finalVal);
                Pack128 vv;
                MULTI128<weightUpdate<float>, float>().weightUpdate(weightVal, finalVal, vv);
              
                Store128(wPack+wOffset, vv);
                float4 wf4 = *(reinterpret_cast<float4*>(&vv));
                float4 rLambdaWF4 = *(reinterpret_cast<float4*>(&rLambdaW));
                v[u] = float4ToHalf4(vv);


              }
            }

            if (!flagThread) {
              int wOffset = (u)*(WARP_SIZE-2) + 1 - (2*wid)/NCCL_LL128_LINEELEMS;

              totalOffset = sumOffset + (wOffset)*(sizeof(Pack128)/sizeof(float));
              
              if (totalOffset < totalSize) {
                if (COMPUTE) {
                  Pack128 wght;
                  Fetch128(wght, (wPack+wOffset));
                  Pack128 vval;
                  Fetch128(vval, (vPack+wOffset));
                  Pack128 mval;
                  Fetch128(mval, (mPack+wOffset));

                  Pack128 S5;
                  uint2 __val = *(reinterpret_cast<const uint2*>(&v[u+1]));
                  Pack128 S4, S3, S2, S1;

                  MULTI128<mixedbinOp1<float>, half>().mixedbinOp1(beta2, vval, __val, S2);
                  MULTI128<binOp2<float>, float>().binOp2(beta2, S2, S4);

                  MULTI128<mixedbinOp3<float>, half>().mixedbinOp1(beta2, mval, __val, S1);
                  MULTI128<binOp2<float>, float>().binOp4(beta1, S1, S3);
                  
                  Store128(vPack+wOffset, S2);
                  Store128(mPack+wOffset, S1);

                  Pack128 r;
                  MULTI128<rOp<float>, float>().r(wght, S3, S4, r);
                  Store128(rPack+wOffset, r);

                  float4 f4 = *(reinterpret_cast<float4*>(&wght));
                  *wVal += f4.x*f4.x + f4.y*f4.y + f4.z*f4.z + f4.w*f4.w;

                  f4 = *(reinterpret_cast<float4*>(&r));
                  *rVal += f4.x*f4.x + f4.y*f4.y + f4.x*f4.x + f4.y*f4.y;
                } else {
                  Pack128 rLambdaW;
                  Fetch128(rLambdaW, rPack+wOffset);

                  Pack128 weightVal;
                  Fetch128(weightVal, wPack+wOffset);

                  Pack128 finalVal;
                  MULTI128<delta<float>, float>().delta(lr*scale, rLambdaW, finalVal);
                  Pack128 vv;
                  MULTI128<weightUpdate<float>, float>().weightUpdate(weightVal, finalVal, vv);
                
                  Store128(wPack+wOffset, vv);
                  float4 wf4 = *(reinterpret_cast<float4*>(&vv));
                  v[u+1] = float4ToHalf4(vv);
                }
              }
            }
          }
        
          if (DST) {
            shmem64Ptr[u*(WARP_SIZE-2)] = v[u];
            if (!flagThread) shmem64Ptr[u*(WARP_SIZE-2)+1] = v[u+1];
          }
        }
      }
      /*********** End data Storing : REG -> SHMEM ************/

      
      /************************ Send **************************/
      if (SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) {
          int flag = sendFlag(i);
          uint64_t* ptr = sendPtr(i)+ll128Offset;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
          }
        }
        int flag = sendFlag(0);
        uint64_t* ptr = sendPtr(0)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
        }
      }
      /********************** End Send ************************/
    }


    template <class FUNC1, typename T1, int ELEMS_PER_THREAD, int RECV, int SEND, int SRC, int DST, int COMPUTE>
    __device__ __forceinline__ void recvReduceSendCopy(int ll128Offset) {
      uint64_t v[ELEMS_PER_THREAD];

      /************* Data Loading : SHMEM -> REG **************/
      if (SRC) {
        volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          v[u] = shmem64Ptr[u*(WARP_SIZE-2)];
          if (!flagThread) v[u+1] = shmem64Ptr[u*(WARP_SIZE-2)+1];
        }
      }
      /*********** End Data Loading : SHMEM -> REG ************/

      /************************ Recv **************************/
      if (RECV) {
        uint64_t flag = recvFlag(0);
        uint64_t* ptr = recvPtr(0)+ll128Offset;
        bool needReload;
        uint64_t v0, v1;
        do {
          needReload = false;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, v0, v1);
            needReload |= flagThread && (v1 != flag);
          }
        } while (__any_sync(WARP_MASK, needReload) && checkAbort(0, 0) == 0);
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          load128(ptr+u*WARP_SIZE, v0, v1);
          v[u] = SRC ? MULTI<FUNC1, T1>()(v0, v[u]) : v0;
          v[u+1] = SRC ? MULTI<FUNC1, T1>()(v1, v[u+1]) : v1;
        }

        for (int i=1; i<NRECV && i<nrecv; i++) {
          uint64_t flag = recvFlag(i);
          uint64_t* ptr = recvPtr(i)+ll128Offset;
          uint64_t v0, v1;
          do {
            needReload = false;
            #pragma unroll
            for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
              load128(ptr+u*WARP_SIZE, v0, v1);
              needReload |= flagThread && (v1 != flag);
            }
          } while (__any_sync(WARP_MASK, needReload) && checkAbort(i, 0) == 0);
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            load128(ptr+u*WARP_SIZE, v0, v1);
            v[u] = MULTI<FUNC1, T1>()(v0, v[u]);
            v[u+1] = MULTI<FUNC1, T1>()(v1, v[u+1]);
          }
        }
      }
      /********************** End Recv ************************/

      /************************ Send **************************/
      if (SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) {
          int flag = sendFlag(i);
          uint64_t* ptr = sendPtr(i)+ll128Offset;
          #pragma unroll
          for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
            store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
          }
        }
        int flag = sendFlag(0);
        uint64_t* ptr = sendPtr(0)+ll128Offset;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          store128(ptr+u*WARP_SIZE, v[u], flagThread ? flag : v[u+1]);
        }
      }
      /********************** End Send ************************/

      /************* Data Storing : REG -> SHMEM **************/
      if (DST) {
        volatile uint64_t* shmem64Ptr = shmem - (2*wid)/NCCL_LL128_LINEELEMS;
        #pragma unroll
        for (int u=0; u<ELEMS_PER_THREAD; u+=2) {
          shmem64Ptr[u*(WARP_SIZE-2)] = v[u];
          if (!flagThread) shmem64Ptr[u*(WARP_SIZE-2)+1] = v[u+1];
        }
      }
      /*********** End data Storing : REG -> SHMEM ************/
    }

    #define LL128INC (WARP_SIZE*NCCL_LL128_SHMEM_ELEMS_PER_THREAD)
    #define ELEMINC (LL128INC-(LL128INC/NCCL_LL128_LINEELEMS))


    template <int RECV, int SEND, int SRC, int DST, int COMPUTE, int LAMB_SEND_COMPUTE>
    __device__ void GenericOp2(float lr, float beta1, float beta2, const T* g, T* halfw, float* w, float* m, float* v, float* r,
                              size_t startOffset, int partNum, int nelem, float* wNorm, float* rNorm, float scale) {
      if (nelem <= 0) {
        // Don't move any data but still increase steps and sync with prev/next
        if (SEND) waitSend(0);
        FOR_SEND(incSend); if (SEND) postSend();
        FOR_RECV(incRecv); if (RECV) postRecv();
        return;
      }
      const int nelem64 = ((nelem*sizeof(T))/(2*sizeof(uint64_t)))*2;
      uint64_t* gPack = (uint64_t*)g;
      uint64_t* halfwPack = (uint64_t*)halfw;
      Pack128* wPack = (Pack128*)w;
      Pack128* mPack = (Pack128*)m;
      Pack128* vPack = (Pack128*)v;
      Pack128* rPack = (Pack128*)r;

      int ll128Offset = LL128INC*warp+2*wid;
      int elemOffset = ELEMINC*warp;
      const int nwarps = nthreads/WARP_SIZE;

      if (SEND) waitSend(DIVUP(nelem*sizeof(T), ELEMINC*sizeof(uint64_t))*LL128INC*sizeof(uint64_t));
      barrier();

      while (elemOffset*(sizeof(uint64_t)/sizeof(T)) < nelem) {
        const int maxOffset128 = min(nelem64-elemOffset, (int)ELEMINC);
        const int maxOffset = min(nelem-(elemOffset*((int)(sizeof(uint64_t)/sizeof(T)))), (int)(ELEMINC*(sizeof(uint64_t)/sizeof(T))));
        if (SRC) {
          int done = 0;
          if ((((uint64_t)g)&0xf) == 0) {
            loadSrcToShmem128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, gPack+elemOffset+2*wid);
            done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
          }
          loadSrcToShmem(done, maxOffset, (T*)(gPack+elemOffset));
        }
        __syncwarp();

        recvReduceSendCopy2<NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SRC, DST, COMPUTE, LAMB_SEND_COMPUTE>(lr, beta1, beta2, gPack, 
                                                                                                                halfwPack+elemOffset+2*wid, 
                                                                                                                wPack+elemOffset+2*wid, 
                                                                                                                  mPack+elemOffset+2*wid, vPack+elemOffset+2*wid, rPack+elemOffset+2*wid, startOffset+(elemOffset+2*wid)*sizeof(Pack128)/sizeof(float), ll128Offset, 
                                                                                                                  wNorm, rNorm, scale);
        __syncwarp();
        if (DST) {
          int done = 0;
          if ((((uint64_t)w)&0xf) == 0) {
            storeShmemToDst128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, halfwPack+elemOffset+2*wid);
            done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
          }
          storeShmemToDst(done, maxOffset, (T*)(halfwPack+elemOffset));
        }
        __syncwarp();
        ll128Offset += LL128INC*nwarps;
        elemOffset += ELEMINC*nwarps;
      }

      barrier();
      FOR_SEND(incSend); if (SEND) postSend();
      FOR_RECV(incRecv); if (RECV) postRecv();
    }

    template <int RECV, int SEND, int SRC, int DST>
    __device__ void GenericOp(const T* srcPtr, T* dstPtr, int nelem) {
      if (nelem <= 0) {
        // Don't move any data but still increase steps and sync with prev/next
        if (SEND) waitSend(0);
        FOR_SEND(incSend); if (SEND) postSend();
        FOR_RECV(incRecv); if (RECV) postRecv();
        return;
      }
      const int nelem64 = ((nelem*sizeof(T))/(2*sizeof(uint64_t)))*2;
      const uint64_t* src64Ptr = ((uint64_t*)srcPtr);
      uint64_t* dst64Ptr = ((uint64_t*)dstPtr);

      int ll128Offset = LL128INC*warp+2*wid;
      int elemOffset = ELEMINC*warp;
      const int nwarps = nthreads/WARP_SIZE;

      if (SEND) waitSend(DIVUP(nelem*sizeof(T), ELEMINC*sizeof(uint64_t))*LL128INC*sizeof(uint64_t));
      barrier();

      while (elemOffset*(sizeof(uint64_t)/sizeof(T)) < nelem) {
        const int maxOffset128 = min(nelem64-elemOffset, (int)ELEMINC);
        const int maxOffset = min(nelem-(elemOffset*((int)(sizeof(uint64_t)/sizeof(T)))), (int)(ELEMINC*(sizeof(uint64_t)/sizeof(T))));
        if (SRC) {
          int done = 0;
          if ((((uint64_t)srcPtr)&0xf) == 0) {
            loadSrcToShmem128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, src64Ptr+elemOffset+2*wid);
            done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
          }
          loadSrcToShmem(done, maxOffset, (T*)(src64Ptr+elemOffset));
        }
        __syncwarp();
        recvReduceSendCopy<FUNC, T, NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SRC, DST, 0>(ll128Offset);
        __syncwarp();
        if (DST) {
          int done = 0;
          if ((((uint64_t)dstPtr)&0xf) == 0) {
            storeShmemToDst128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, dst64Ptr+elemOffset+2*wid);
            done = maxOffset128*(sizeof(uint64_t)/sizeof(T));
          }
          storeShmemToDst(done, maxOffset, (T*)(dst64Ptr+elemOffset));
        }
        __syncwarp();
        ll128Offset += LL128INC*nwarps;
        elemOffset += ELEMINC*nwarps;
      }

      barrier();
      FOR_SEND(incSend); if (SEND) postSend();
      FOR_RECV(incRecv); if (RECV) postRecv();
    }



    template <int RECV, int SEND, int SRC, int DST>
    __device__ void GenericOpF32(const float* srcPtr, float* dstPtr, int nelem) {
      if (nelem <= 0) {
        // Don't move any data but still increase steps and sync with prev/next
        if (SEND) waitSend(0);
        FOR_SEND(incSend); if (SEND) postSend();
        FOR_RECV(incRecv); if (RECV) postRecv();
        return;
      }
      const int nelem64 = ((nelem*sizeof(float))/(2*sizeof(uint64_t)))*2;
      const uint64_t* src64Ptr = ((uint64_t*)srcPtr);
      uint64_t* dst64Ptr = ((uint64_t*)dstPtr);

      int ll128Offset = LL128INC*warp+2*wid;
      int elemOffset = ELEMINC*warp;
      const int nwarps = nthreads/WARP_SIZE;

      if (SEND) waitSend(DIVUP(nelem*sizeof(float), ELEMINC*sizeof(uint64_t))*LL128INC*sizeof(uint64_t));
      barrier();

      while (elemOffset*(sizeof(uint64_t)/sizeof(float)) < nelem) {
        const int maxOffset128 = min(nelem64-elemOffset, (int)ELEMINC);
        const int maxOffset = min(nelem-(elemOffset*((int)(sizeof(uint64_t)/sizeof(float)))), (int)(ELEMINC*(sizeof(uint64_t)/sizeof(float))));
        if (SRC) {
          int done = 0;
          if ((((uint64_t)srcPtr)&0xf) == 0) {
            loadSrcToShmem128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, src64Ptr+elemOffset+2*wid);
            done = maxOffset128*(sizeof(uint64_t)/sizeof(float));
          }
          loadSrcToShmem(done, maxOffset, (float*)(src64Ptr+elemOffset));
        }
        __syncwarp();
        recvReduceSendCopy<FuncSum<float>, float, NCCL_LL128_SHMEM_ELEMS_PER_THREAD, RECV, SEND, SRC, DST, 0>(ll128Offset);
        __syncwarp();
        if (DST) {
          int done = 0;
          if ((((uint64_t)dstPtr)&0xf) == 0) {
            storeShmemToDst128<NCCL_LL128_SHMEM_ELEMS_PER_THREAD>(maxOffset128-2*wid, dst64Ptr+elemOffset+2*wid);
            done = maxOffset128*(sizeof(uint64_t)/sizeof(float));
          }
          storeShmemToDst(done, maxOffset, (float*)(dst64Ptr+elemOffset));
        }
        __syncwarp();
        ll128Offset += LL128INC*nwarps;
        elemOffset += ELEMINC*nwarps;
      }

      barrier();
      FOR_SEND(incSend); if (SEND) postSend();
      FOR_RECV(incRecv); if (RECV) postRecv();
    }

    __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
      recvBuff[i] = conn->ll128Buff;
      recvStep[i] = conn->step;
      if (wid == i) recvConn = conn;
      nrecv++;
    }
    __device__ __forceinline__ void loadRecvSync() {
      if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
        recvConnHeadPtr = recvConn->head;
        recvConnHead = recvConn->step;
        // Update opCount in case we skipped some operations
        *(recvConn->opCountLoc) = opCount;
      }
    }

    __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i) {
      sendBuff[i] = conn->ll128Buff;
      sendStep[i] = conn->step;
      if (wid == i) sendConn = conn;
      nsend++;
    }
    __device__ __forceinline__ void loadSendSync() {
      if (tid < nsend) {
        sendConnHeadPtr = sendConn->head;
        sendConnHeadCache = *sendConnHeadPtr;
        sendConnHead = sendConn->step;
        sendConnFifoPtr = sendConn->fifo;
        *(sendConn->opCountLoc) = opCount;
      }
      if (tid >= nthreads-WARP_SIZE && wid<nsend) {
        if (sendConn->fifo) {
          sendConnTailPtr = sendConn->tail;
          sendConnTail = sendConn->step;
        }
      }
    }

    __device__ __forceinline__ void saveRecvSync() {
      if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
        recvConn->step = recvConnHead;
        *(recvConn->opCountLoc) = opCount+1;
        __threadfence_block();
      }
    }

    __device__ __forceinline__ void saveSendSync() {
      if (tid < nsend) {
        sendConn->step = sendConnHead;
        *(sendConn->opCountLoc) = opCount+1;
        __threadfence_block();
      }
    }

  public:
    __device__ __forceinline__
    ncclLL128PrimitivesComputation(const int tid, const int nthreads, int* recvPeers, int* sendPeers, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount, const size_t totalSize, const size_t nranks, const size_t nChannels, const ssize_t loopSize, const ssize_t maxPartSize)
      : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), warp(tid/WARP_SIZE), flagThread((tid%8)==7), opCount(opCount), shmem(ncclShmem+(threadIdx.x/WARP_SIZE)*NCCL_LL128_SHMEM_ELEMS_PER_THREAD*WARP_SIZE+2*wid),
      totalSize(totalSize), nChannels(nChannels), nranks(nranks) , loopSize(loopSize), maxPartSize(maxPartSize) {
      // Make sure step is updated before we read it.
      barrier();

      for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);
      for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
      loadRecvSync();
      loadSendSync();
    }

    __device__ void send(const T* src, int nelem) {
      return GenericOp<0, 1, 1, 0>(src, NULL, nelem);
    }

    __device__ void sendLAMB(float lr, const T* src, T* halfw, float* w, float* r, int nelem, int offset, float scale) {
      return GenericOp2<0, 1, 1, 1, 0, 1>(lr, 0,0, src, halfw, w, nullptr, nullptr, r, offset, 0, nelem, nullptr, nullptr, scale);
    }
    
    __device__ void recv(T* dst, int nelem) {
      return GenericOp<1, 0, 0, 1>(NULL, dst, nelem);
    }

    __device__ void recvAllGatherCompute(/*PRIMSLL128: {INSERT recv ARGS}*/ int nelem) {
      return GenericOp2<1, 0, 0, 1, 0, 1>(/*{INSERT GenericOp2 CALL PARAMS for ALLGATHER COMPUTE}*/ nelem);
    }

    __device__ void recvReduceSend(const T* src, int nelem) {
      return GenericOp<1, 1, 1, 0>(src, NULL, nelem);
    }

    __device__ void recvReduceCopy(const T* src, T* dst, int nelem) {
      return GenericOp<1, 0, 1, 1>(src, dst, nelem);
    }

    __device__ void recvReduceCopy2(float lr, float beta1, float beta2, const T* g, T* halfw, float* w, 
                          float* m, float* v, float* r, int offset, int partNum, int nelem, float* wNorm, float* rNorm) {
      return GenericOp2<1, 0, 1, 0, 1, 0>(lr, beta1, beta2, g, halfw, w, m, v, r, offset, partNum, nelem, wNorm, rNorm, 0);
    }

    __device__ void recvReduceCopyAllGatherCompute(/*PRIMSLL128: {INSERT recv ARGS}*/ int nelem) {
      return GenericOp2<1, 0, 1, 1, 0, 1>(/*{INSERT GenericOp2 CALL PARAMS for ALLGATHER COMPUTE}*/ nelem);
    }

    __device__ void copySend(const T* src, T* dst, int nelem) {
      return GenericOp<0, 1, 1, 1>(src, dst, nelem);
    }

    __device__ void recvCopySend(T* dst, int nelem) {
      return GenericOp<1, 1, 0, 1>(NULL, dst, nelem);
    }

    __device__ void recvCopySendAllGatherCompute(/*PRIMSLL128: {INSERT recv ARGS}*/ int nelem) {
      return GenericOp2<1, 1, 0, 1, 0, 1>(/*{INSERT GenericOp2 CALL PARAMS for ALLGATHER COMPUTE}*/nelem);
    }


    __device__ void recvReduceCopySend(T lr, T beta1, T beta2, const T* g, T* w, T* m, T* v, size_t offset, int partNum, int nelem) {
      return GenericOp2<1, 1, 1, 1, 1, 0>(lr, beta1, beta2, g, w, m, v, offset, partNum, nelem);
    }

    __device__ void sendF32(const float* src, int nelem) {
      return GenericOpF32<0, 1, 1, 0>(src, NULL, nelem);
    }

    __device__ void recvReduceSendF32(const float* src, int nelem) {
      return GenericOpF32<1, 1, 1, 0>(src, NULL, nelem);
    }

    __device__ void recvReduceCopyF32(const float* src, float* dst, int nelem) {
      return GenericOpF32<1, 0, 1, 1>(src, dst, nelem);
    }

    __device__ void recvReduceCopySendF32(const float* src, float* dst, int nelem) {
      return GenericOpF32<1, 1, 1, 1>(src, dst, nelem);
    }

    __device__ void recvF32(float* dst, int nelem) {
      return GenericOpF32<1, 0, 0, 1>(NULL, dst, nelem);
    }

    __device__ void recvCopySendF32(float* dst, int nelem) {
      return GenericOpF32<1, 1, 0, 1>(NULL, dst, nelem);
    }

    __device__ __forceinline__ ~ncclLL128PrimitivesComputation() {
      // Save steps for the next operation
      saveRecvSync();
      saveSendSync();
    }
  };
#endif