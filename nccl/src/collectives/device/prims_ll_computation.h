#define TYPE_PRIMS_LL 1





#if TYPE_PRIMS_LL == 0

template <typename TF16, class FUNCF16, typename TF32, typename FUNCF32, int NRECV, int NSEND>
class ncclLLPrimitivesComputation {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  union ncclLLFifoLine* recvBuff[NRECV];
  union ncclLLFifoLine* sendBuff[NSEND];
  struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*NCCL_LL_SLICE_LINES; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*NCCL_LL_SLICE_LINES; }
  inline __device__ ncclLLFifoLine* recvPtr(int i) { return (ncclLLFifoLine*)((char*)(recvBuff[i]+recvOffset(i)));} //TODO: see this
  inline __device__ ncclLLFifoLine* sendPtr(int i) { return (ncclLLFifoLine*)((char*)(sendBuff[i]+sendOffset(i)));}
  inline __device__ uint32_t recvFlag(int i) { return NCCL_LL_FLAG(recvStep[i]+1); }
  inline __device__ uint32_t sendFlag(int i) { return NCCL_LL_FLAG(sendStep[i]+1); }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
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
        int size = ((sendConnHead & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) ? NCCL_LL_SLICE_LINES*sizeof(union ncclLLFifoLine) : nbytes;
        sendConnFifoPtr[sendConnHead%NCCL_STEPS] = size;
      }
      sendConnHead += 1;
    }
    barrier();
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += 1;
  }
  inline __device__ void postRecv() {
    barrier();
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
  }

  inline __device__ void incSend(int i, int offset) {
    // LL Cleanup : write all flags in the slice to make sure we don't have
    // data corruption when flag loops over.
    if ((sendStep[i] & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
      for (int o = offset; o<NCCL_LL_SLICE_LINES; o+=nthreads) storeLL(sendPtr(i)+o, 0, sendFlag(i));
    }
    sendStep[i]++;
  }

  // __device__ uint32_t readLLTF16(int i, int offset) {
  //   union ncclLLFifoLineHalf* src = recvPtr(i) + offset;
  //   uint32_t flag = recvFlag(i); //TODO: check the recvFlag
  //   uint32_t data_, flag_;
  //   spins = 0;
  //   mismatch = 0;
  //   do {
  //     asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(data_), "=r"(flag_) : "l"(&src->i2));
  //     if (checkAbort(i, 0)) break;
  //   } while (flag_ != flag);
  //   return data_;
  // }

  __device__ uint64_t readLL(int i, int offset) {
    union ncclLLFifoLine* src = recvPtr(i) + offset;
    uint32_t flag = recvFlag(i);
    uint32_t data1, flag1, data2, flag2;
    spins = 0;
    mismatch = 0;
    do {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2) : "l"(&src->i4));
      if (checkAbort(i, 0)) break;
    } while ((flag1 != flag) || (flag2 != flag));
    uint64_t val64 = data1 + (((uint64_t)data2) << 32);
    return val64;
  }

  __device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)), "r"(flag));
  }

  // __device__ void storeLL(union ncclLLFifoLineHalf* dst, uint32_t val, uint32_t flag) {
  //   asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" :: "l"(&dst->i2), "r"(val), "r"(flag));
  // }

  // Using memcpy handles misaligned pointers.
  __device__ uint64_t readAL(uint64_t* src) {
    uint64_t val;
    memcpy((char*)&val, (char*)src, sizeof(uint64_t));
    return val;
  }

  // __device__ uint64_t readAL(uint32_t* src) {
  //   uint32_t val;
  //   memcpy((char*)&val, (char*)src, sizeof(uint32_t));
  //   return val;
  // }

  __device__ void storeAL(uint64_t* dst, uint64_t val, uint32_t nbytes) {
    memcpy((char*)dst, (char*)&val, nbytes);
  }

  // __device__ void storeAL(uint32_t* dst, uint32_t val, uint32_t nbytes) {
  //   memcpy((char*)dst, (char*)&val, nbytes);
  // }
  
  __device__ uint32_t SumF16(uint32_t val1, uint32_t val2) {
    half2 val1Half2 = *(reinterpret_cast<half2*>(&val1));
    half2 val2Half2 = *(reinterpret_cast<half2*>(&val2));

    half2 finalVal = __hadd2(val1Half2, val2Half2);

    return *(reinterpret_cast<uint32_t*> (&finalVal));
  }

  // __device__ uint64_t SumF16(uint64_t val1, uint32_t val2) {
  //   half2 val1Half2 = *(reinterpret_cast<half2*>(&val1));
  //   half2 val2Half2 = *(reinterpret_cast<half2*>(&val2));

  //   half2 finalVal = __hadd2(val1Half2, val2Half2);

  //   return *(reinterpret_cast<uint32_t*> (&finalVal));
  // }
  
  __device__ uint64_t half2ToFloat2(uint32_t val) {
    half2 val1Half2 = *(reinterpret_cast<half2*>(&val));
    float2 f2 = __half22float2(val1Half2);
    // if (threadIdx.x == 0) {
    //   printf("x %f y %f\n", f2.x, f2.y);
    // }

    return *(reinterpret_cast<uint64_t*>(&f2));
  }

  template <int RECV, int SEND, int SRC, int DST, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ void LLGenericOp2(TF32 lr, TF32 beta1, TF32 beta2, const TF16* g, TF16* halfw, TF32* w, TF32* m, TF32* v,  int nelem) {
    uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(TF16);
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    uint64_t* gPack = (uint64_t*)g;
    uint64_t* halfwPack = (uint64_t*)halfw;
    ulong2* wPack = (ulong2*)w;
    ulong2* mPack = (ulong2*)m;
    ulong2* vPack = (ulong2*)v;


    int offset = tid;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(npack*sizeof(union ncclLLFifoLine));

    // Do multiples of 64 bits
    #pragma unroll 2
    for (; offset<npack; offset+=nthreads) {
      // Recv : local, then intra-node, then inter-node
      uint64_t val = SRC ? readAL(gPack+offset) : readLL(0, offset);
      if (RECV) {
        if (SRC) val = MULTI<FuncSum<TF16>, TF16>()(readLL(0, offset), val);
        for (int i=1; i<NRECV && i<nrecv; i++) {
          val = MULTI<FuncSum<TF16>, TF16>()(readLL(i, offset), val);
        }
      }

      // Send : inter-node, then intra-node, then local
      if(COMPUTE) {
        Pack128 readVal;
        Fetch128(readVal, (wPack+offset));
        Pack128 vval;
        Fetch128(vval, (vPack+offset));
        Pack128 mval;
        Fetch128(mval, (mPack+offset));

        Pack128 S5;
        uint2 __val = *(reinterpret_cast<const uint2*>(&val));
        Pack128 S4, S3, S2, S1;

        MULTI128<mixedbinOp1<float>, half>().mixedbinOp1(beta2, vval, __val, S2);
        MULTI128<binOp2<float>, half>().binOp2(beta2, S2, S4);

        MULTI128<mixedbinOp3<float>, half>().mixedbinOp1(beta2, mval, __val, S1);
        MULTI128<binOp2<float>, half>().binOp4(beta1, S1, S3);
        
        MULTI128<delta<float>, half>().delta(lr, S3, S4, S5);

        Store128(vPack+offset, S2);
        Store128(mPack+offset, S1);

        Pack128 finalVal;
        MULTI128<weightUpdate<float>, half>().weightUpdate(readVal, S5, finalVal);
        *(wPack+offset) = finalVal;
        uint64_t fp16FinalVal = float4ToHalf4(finalVal);
        
        storeAL(halfwPack+offset, fp16FinalVal, sizeof(uint64_t));

        if (SEND) {
          for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, fp16FinalVal, sendFlag(i));
          storeLL(sendPtr(0)+offset, fp16FinalVal, sendFlag(0));
        }
      }
    }

    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  template <typename T, int RECV, int SEND, int SRC, int DST>
  __device__ void LLGenericOp(const T* srcPtr, T* dstPtr, int nelem) {
    uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(T);
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    uint64_t* srcPack = (uint64_t*)srcPtr;
    uint64_t* dstPack = (uint64_t*)dstPtr;
    int offset = tid;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(npack*sizeof(union ncclLLFifoLine));

    // Do multiples of 64 bits
    #pragma unroll 2
    for (; offset<npack; offset+=nthreads) {
      // Recv : local, then intra-node, then inter-node
      uint64_t val = SRC ? readAL(srcPack+offset) : readLL(0, offset);
      if (RECV) {
        if (SRC) val = MULTI<FuncSum<T>, T>()(readLL(0, offset), val);
        for (int i=1; i<NRECV && i<nrecv; i++) {
          val = MULTI<FuncSum<T>, T>()(readLL(i, offset), val);
        }
      }

      // Send : inter-node, then intra-node, then local
      if (SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, val, sendFlag(i));
        storeLL(sendPtr(0)+offset, val, sendFlag(0));
      }
      if (DST) {
        if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {
          // Last incomplete word
          storeAL(dstPack+offset, val, nbytes & 0x7);
        } else {
          storeAL(dstPack+offset, val, sizeof(uint64_t));
        }
      }
    }
    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = conn->llBuff;
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
    sendBuff[i] = conn->llBuff;
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
  ncclLLPrimitivesComputation(const int tid, const int nthreads, int* recvPeers, int* sendPeers, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), opCount(opCount) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
    loadRecvSync();
    loadSendSync();
  }

  //F16 functions
  __device__ void send(const TF16* src, int nelem) {
    return LLGenericOp<TF16, 0, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recv(TF16* dst, int nelem) {
    return LLGenericOp<TF16, 1, 0, 0, 1>(NULL, dst, nelem);
  }
 
  __device__ void recvReduceSend(const TF16* src, int nelem) {
    return LLGenericOp<TF16, 1, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recvReduceCopy(const TF16* src, TF16* dst, int nelem) {
    return LLGenericOp<TF16, 1, 0, 1, 1>(src, dst, nelem);
  }

  __device__ void copySend(const TF16* src, TF16* dst, int nelem) {
    return LLGenericOp<TF16, 0, 1, 1, 1>(src, dst, nelem);
  }

  __device__ void recvCopySend(TF16* dst, int nelem) {
    return LLGenericOp<TF16, 1, 1, 0, 1>(NULL, dst, nelem);
  }

  //F32 functions
  __device__ void sendF32(const TF32* src, int nelem) {
    return LLGenericOp<TF32, 0, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recvF32(TF32* dst, int nelem) {
    return LLGenericOp<TF32, 1, 0, 0, 1>(NULL, dst, nelem);
  }
 
  __device__ void recvReduceSendF32(const TF32* src, int nelem) {
    return LLGenericOp<TF32, 1, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recvReduceCopyF32(const TF32* src, TF32* dst, int nelem) {
    return LLGenericOp<TF32, 1, 0, 1, 1>(src, dst, nelem);
  }

  __device__ void copySendF32(const TF32* src, TF32* dst, int nelem) {
    return LLGenericOp<TF32, 0, 1, 1, 1>(src, dst, nelem);
  }

  __device__ void recvCopySendF32(TF32* dst, int nelem) {
    return LLGenericOp<TF32, 1, 1, 0, 1>(NULL, dst, nelem);
  }

  __device__ void recvReduceCopySend(TF32 lr, TF32 beta1, TF32 beta2, const TF16* g, TF16* halfw, TF32* w, TF32* m, TF32* v,  int nelem) {
    return LLGenericOp2<1, 1, 1, 1, 1, 0>(lr, beta1, beta2, g, halfw, w, m, v, nelem);
  }

  __device__ __forceinline__ ~ncclLLPrimitivesComputation() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};
#elif TYPE_PRIMS_LL == 1
template <typename T, class FUNC, int NRECV, int NSEND>
class ncclLLPrimitivesComputation {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int nChannels;
  const ssize_t maxPartSize;
  const size_t loopSize;
  int nrecv = 0;
  int nsend = 0;
  size_t totalSize;
  size_t nranks;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  union ncclLLFifoLine* recvBuff[NRECV];
  union ncclLLFifoLine* sendBuff[NSEND];
  struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*NCCL_LL_SLICE_LINES; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*NCCL_LL_SLICE_LINES; }
  inline __device__ union ncclLLFifoLine* recvPtr(int i) { return recvBuff[i]+recvOffset(i); }
  inline __device__ union ncclLLFifoLine* sendPtr(int i) { return sendBuff[i]+sendOffset(i); }
  inline __device__ uint32_t recvFlag(int i) { return NCCL_LL_FLAG(recvStep[i]+1); }
  inline __device__ uint32_t sendFlag(int i) { return NCCL_LL_FLAG(sendStep[i]+1); }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
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
        int size = ((sendConnHead & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) ? NCCL_LL_SLICE_LINES*sizeof(union ncclLLFifoLine) : nbytes;
        sendConnFifoPtr[sendConnHead%NCCL_STEPS] = size;
      }
      sendConnHead += 1;
    }
    barrier();
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += 1;
  }
  inline __device__ void postRecv() {
    barrier();
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += 1;
  }

  inline __device__ void incSend(int i, int offset) {
    // LL Cleanup : write all flags in the slice to make sure we don't have
    // data corruption when flag loops over.
    if ((sendStep[i] & NCCL_LL_CLEAN_MASK) == NCCL_LL_CLEAN_MASK) {
      for (int o = offset; o<NCCL_LL_SLICE_LINES; o+=nthreads) storeLL(sendPtr(i)+o, 0, sendFlag(i));
    }
    sendStep[i]++;
  }

  __device__ uint64_t readLL(int i, int offset) {
    union ncclLLFifoLine* src = recvPtr(i) + offset;
    uint32_t flag = recvFlag(i);
    uint32_t data1, flag1, data2, flag2;
    spins = 0;
    mismatch = 0;
    do {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2) : "l"(&src->i4));
      if (checkAbort(i, 0)) break;
    } while ((flag1 != flag) || (flag2 != flag));
    uint64_t val64 = data1 + (((uint64_t)data2) << 32);
    return val64;
  }

  __device__ void storeLL(union ncclLLFifoLine* dst, uint64_t val, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" :: "l"(&dst->i4), "r"((uint32_t)val), "r"(flag), "r"((uint32_t)(val >> 32)), "r"(flag));
  }

  // Using memcpy handles misaligned pointers.
  __device__ uint64_t readAL(uint64_t* src) {
    uint64_t val;
    memcpy((char*)&val, (char*)src, sizeof(uint64_t));
    return val;
  }

  __device__ void storeAL(uint64_t* dst, uint64_t val, uint32_t nbytes) {
    memcpy((char*)dst, (char*)&val, nbytes);
  }

    template <int RECV, int SEND, int SRC, int DST, int LAMB_SEND_COMPUTE>
  __device__ void LLGenericOp3(float lr, const T* g, T* halfw, float* w, float* r, int nelem, int partNum, float scale) {
    uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(T);
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    
    uint64_t* halfwPack = (uint64_t*)halfw;
    Pack128* wPack = (Pack128*)w;
    Pack128* rPack = (Pack128*)r;

    // if (COMPUTE && nelem > 0) {
    //   if (threadIdx.x == 0) {
    //       printf("nelem %d totalSize %ld\n", nelem, totalSize);
    //     }
    // }
    int offset = tid;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(npack*sizeof(union ncclLLFifoLine));

    // Do multiples of 64 bits
    #pragma unroll 2
    for (; offset<npack; offset+=nthreads) {
      // Recv : local, then intra-node, then inter-node
      uint64_t val = SRC ? readAL(halfwPack+offset) : readLL(0, offset);
      if (RECV) {
        if (SRC) val = MULTI<FUNC, T>()(readLL(0, offset), val);
        for (int i=1; i<NRECV && i<nrecv; i++) {
          val = MULTI<FUNC, T>()(readLL(i, offset), val);
        }
      }

      if(LAMB_SEND_COMPUTE) {
        // const size_t slicedMomentSize = totalSize/nranks;
        // const int numParts = (maxPartSize > 0) ? max(1UL, slicedMomentSize/maxPartSize) : 1;
        // const int partsPerChannel = (totalSize < loopSize) ? 1 : DIVUP(totalSize, loopSize);
        // const size_t partIdx = blockIdx.x*partsPerChannel + partNum;

        // const size_t partStartOffset = partIdx*maxPartSize;
        // if (threadIdx.x == 0) {
        //   printf("startOffset %ld partStartOffset %ld blockIdx.x %d offset %d maxPartSize %ld partNum %d rank %d\n", startOffset, partStartOffset, blockIdx.x, offset, maxPartSize, partNum, comm->rank);
        // }
        Pack128 rLambdaW;
        Fetch128(rLambdaW, rPack+offset);

        Pack128 weightVal;
        Fetch128(weightVal, wPack+offset);

        Pack128 finalVal;
        MULTI128<delta<float>, float>().delta(lr*scale, rLambdaW, finalVal);
        Pack128 vv;
        MULTI128<weightUpdate<float>, float>().weightUpdate(weightVal, finalVal, vv);
      
        Store128(wPack+offset, vv);
        float4 wf4 = *(reinterpret_cast<float4*>(&vv));
        // if (offset == 0) {
        //   printf("wf4.x %f\n", wf4.x);
        // }
        val = float4ToHalf4(vv);
        if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {
          // Last incomplete word
          storeAL(halfwPack+offset, val, nbytes & 0x7);
        } else {
          storeAL(halfwPack+offset, val, sizeof(uint64_t));
        }
        //uint64_t delta = MULTI<delta<T>, T>().delta(lr*scale, r);
        //val = MULTI<weightUpdate<T>, T>().weightUpdate(weight, delta);
        // float2 f2 = *(reinterpret_cast<float2*>(&val));
        // if (fabs(f2.x - 0.5f)/0.5 <= 1e-5) {
        //   printf("r %f update %f scale %f f2.x %f\n", r, val, scale, f2.x);
        // }
        //static_assert(COMPUTE==1 && DST==1 && SEND==1);
        // if (SEND) {
        //   for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, S5, sendFlag(i));
        //   storeLL(sendPtr(0)+offset, S5, sendFlag(0));
        // }
      }

      // Send : inter-node, then intra-node, then local
      if (LAMB_SEND_COMPUTE && SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, val, sendFlag(i));
        storeLL(sendPtr(0)+offset, val, sendFlag(0));
      }
    }
    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  template <int RECV, int SEND, int SRC, int DST, int COMPUTE, int ALLGATHER_COMPUTE>
  __device__ void LLGenericOp2(float lr, float beta1, float beta2, const T* g, T* halfw, float* w, float* m, float* v, float *r, size_t startOffset, 
                               int nelem, int partNum, float* rrVal, float* wVal) {
    uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(T);
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    uint64_t* gPack = (uint64_t*)g;
    uint64_t* halfwPack = (uint64_t*)halfw;
    Pack128* wPack = (Pack128*)w;
    Pack128* mPack = (Pack128*)m;
    Pack128* vPack = (Pack128*)v;
    Pack128* rPack = (Pack128*)r;

    // if (COMPUTE && nelem > 0) {
    //   if (threadIdx.x == 0) {
    //       printf("nelem %d totalSize %ld\n", nelem, totalSize);
    //     }
    // }
    int offset = tid;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(npack*sizeof(union ncclLLFifoLine));

    // Do multiples of 64 bits
    #pragma unroll 2
    for (; offset<npack; offset+=nthreads) {
      // Recv : local, then intra-node, then inter-node
      uint64_t val = SRC ? readAL(gPack+offset) : readLL(0, offset);
      if (RECV) {
        if (SRC) val = MULTI<FUNC, T>()(readLL(0, offset), val);
        for (int i=1; i<NRECV && i<nrecv; i++) {
          val = MULTI<FUNC, T>()(readLL(i, offset), val);
        }
      }

      // Send : inter-node, then intra-node, then local
      if (!COMPUTE && SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, val, sendFlag(i));
        storeLL(sendPtr(0)+offset, val, sendFlag(0));
      }
  
      if(COMPUTE && DST) {
        // if (threadIdx.x == 0) {
        //   printf("startOffset %ld partStartOffset %ld blockIdx.x %d offset %d maxPartSize %ld partNum %d rank %d\n", startOffset, partStartOffset, blockIdx.x, offset, maxPartSize, partNum, comm->rank);
        // }
        uint2 __val = *(reinterpret_cast<const uint2*>(&val));
        Pack128 weight;
        Fetch128(weight, (wPack+offset));
        Pack128 vval;
        Fetch128(vval, (vPack+offset));
        Pack128 mval;
        Fetch128(mval, (mPack+offset));

        Pack128 S5, S4, S3, S2, S1;

        MULTI128<mixedbinOp1<float>, half>().mixedbinOp1(beta2, vval, __val, S2);
        MULTI128<binOp2<float>, float>().binOp2(beta2, S2, S4);

        MULTI128<mixedbinOp3<float>, half>().mixedbinOp1(beta2, mval, __val, S1);
        MULTI128<binOp4<float>, float>().binOp4(beta1, S1, S3);
        
        MULTI128<rOp<float>, float>().r(weight, S3, S4, S5);

        Store128(vPack+offset, S2);
        Store128(mPack+offset, S1);

      
        float4 f4 = *(reinterpret_cast<float4*>(&weight));
        *wVal += f4.x*f4.x + f4.y*f4.y + f4.z*f4.z + f4.w*f4.w;

        f4 = *(reinterpret_cast<float4*>(&S5));
        *rrVal += f4.x*f4.x + f4.y*f4.y + f4.z*f4.z + f4.w*f4.w;
        Store128(rPack+offset, S5);
      }
    }
    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  template <int RECV, int SEND, int SRC, int DST>
  __device__ void LLGenericOpF32(const float* srcPtr, float* dstPtr, int nelem) {
    uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(float);
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    uint64_t* srcPack = (uint64_t*)srcPtr;
    uint64_t* dstPack = (uint64_t*)dstPtr;
    int offset = tid;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(npack*sizeof(union ncclLLFifoLine));

    // Do multiples of 64 bits
    #pragma unroll 2
    for (; offset<npack; offset+=nthreads) {
      // Recv : local, then intra-node, then inter-node
      uint64_t val = SRC ? readAL(srcPack+offset) : readLL(0, offset);
      if (RECV) {
        if (SRC) val = MULTI<FuncSum<float>, float>()(readLL(0, offset), val);
        for (int i=1; i<NRECV && i<nrecv; i++) {
          val = MULTI<FuncSum<float>, float>()(readLL(i, offset), val);
        }
      }

      // Send : inter-node, then intra-node, then local
      if (SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, val, sendFlag(i));
        storeLL(sendPtr(0)+offset, val, sendFlag(0));
      }
      if (DST) {
        if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {
          // Last incomplete word
          storeAL(dstPack+offset, val, nbytes & 0x7);
        } else {
          storeAL(dstPack+offset, val, sizeof(uint64_t));
        }
      }
    }
    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  template <int RECV, int SEND, int SRC, int DST>
  __device__ void LLGenericOp(const T* srcPtr, T* dstPtr, int nelem) {
    uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(T);
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    uint64_t* srcPack = (uint64_t*)srcPtr;
    uint64_t* dstPack = (uint64_t*)dstPtr;
    int offset = tid;

    // Always waitSend in case of cleanup
    if (SEND) waitSend(npack*sizeof(union ncclLLFifoLine));

    // Do multiples of 64 bits
    #pragma unroll 2
    for (; offset<npack; offset+=nthreads) {
      // Recv : local, then intra-node, then inter-node
      uint64_t val = SRC ? readAL(srcPack+offset) : readLL(0, offset);
      if (RECV) {
        if (SRC) val = MULTI<FUNC, T>()(readLL(0, offset), val);
        for (int i=1; i<NRECV && i<nrecv; i++) {
          val = MULTI<FUNC, T>()(readLL(i, offset), val);
        }
      }

      // Send : inter-node, then intra-node, then local
      if (SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, val, sendFlag(i));
        storeLL(sendPtr(0)+offset, val, sendFlag(0));
      }
      if (DST) {
        if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {
          // Last incomplete word
          storeAL(dstPack+offset, val, nbytes & 0x7);
        } else {
          storeAL(dstPack+offset, val, sizeof(uint64_t));
        }
      }
    }
    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i) {
    recvBuff[i] = conn->llBuff;
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
    sendBuff[i] = conn->llBuff;
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
  ncclLLPrimitivesComputation(const int tid, const int nthreads, int* recvPeers, int* sendPeers, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount, size_t _totalSize, int _nRanks, int nChannels, ssize_t loopSize, ssize_t maxPartSize)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), opCount(opCount), totalSize(_totalSize), nranks(_nRanks), nChannels(nChannels), loopSize(loopSize), maxPartSize(maxPartSize) {
    // Make sure step is updated before we read it.
    barrier();

    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);
    loadRecvSync();
    loadSendSync();
  }

  __device__ void send(const T* src, int nelem) {
    return LLGenericOp<0, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void sendLAMB(float lr, const T* src, T* halfw, float* weight, float* r, int nelem, float scale) {
    return LLGenericOp3<0, 1, 1, 1, 1>(lr, src, halfw, weight, r, nelem, 0, scale);
  }
  
  __device__ void recv(T* dst, int nelem) {
    return LLGenericOp<1, 0, 0, 1>(NULL, dst, nelem);
  }

  // __device__ void recvAllGatherCompute(/*{INSERT recv ARGS}*/ int nelem) {
  //   return LLGenericOp2<1, 0, 0, 1, 0, 1>(/*{INSERT LLGenericOp2 CALL PARAMS FOR recv}*/ nelem);
  // }

  __device__ void recvReduceSend(const T* src, int nelem) {
    return LLGenericOp<1, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recvReduceCopy(const T* src, T* dst, int nelem) {
    return LLGenericOp<1, 0, 1, 1>(src, dst, nelem);
  }

  __device__ void recvReduceCopy2(float lr, float beta1, float beta2, const T* g, T* halfw, float* w, float* m, float* v, float* r, size_t offset, int nelem, float* rrVal, float* wVal) {
    return LLGenericOp2<1, 0, 1, 1, 1, 0>(lr, beta1, beta2, g, halfw, w, m, v, r, offset, nelem, 0, rrVal, wVal);
  }

  __device__ void copySend(const T* src, T* dst, int nelem) {
    return LLGenericOp<0, 1, 1, 1>(src, dst, nelem);
  }

  __device__ void recvCopySend(T* dst, int nelem) {
    return LLGenericOp<1, 1, 0, 1>(NULL, dst, nelem);
  }

  // __device__ void recvCopySendAllGatherCompute(/*{INSERT recv ARGS}*/ int nelem) {
  //   return LLGenericOp2<1, 1, 0, 1, 0, 1>(/*{INSERT LLGenericOp2 CALL PARAMS FOR recv}*/ nelem);
  // }

  __device__ void recvReduceCopySend(T lr, T beta1, T beta2, const T* g, T* w, T* m, T* v, size_t offset, int nelem, int partNum) {
    return LLGenericOp2<1, 1, 1, 1, 1, 0>(lr, beta1, beta2, g, w, m, v, offset, nelem, partNum);
  }

  __device__ void sendF32(const float* src, int nelem) {
    return LLGenericOpF32<0, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recvReduceSendF32(const float* src, int nelem) {
    return LLGenericOpF32<1, 1, 1, 0>(src, NULL, nelem);
  }

  __device__ void recvReduceCopyF32(const float* src, float* dst, int nelem) {
    return LLGenericOpF32<1, 0, 1, 1>(src, dst, nelem);
  }

  __device__ void recvF32(float* dst, int nelem) {
    return LLGenericOpF32<1, 0, 0, 1>(NULL, dst, nelem);
  }

  __device__ void recvReduceCopySendF32(const float* src, float* dst, int nelem) {
    return LLGenericOpF32<1, 1, 1, 1>(src, dst, nelem);
  }

   __device__ void recvCopySendF32(float* dst, int nelem) {
    return LLGenericOpF32<1, 1, 0, 1>(NULL, dst, nelem);
  }

  __device__ __forceinline__ ~ncclLLPrimitivesComputation() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};

#endif