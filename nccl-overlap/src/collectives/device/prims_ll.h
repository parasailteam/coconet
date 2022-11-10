template <typename T, class FUNC, int NRECV, int NSEND>
class ncclLLPrimitives {
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

  template <int RECV, int SEND, int SRC, int DST, int WEIGHT_UPDATE, int SEND_UPDATE>
  __device__ void LLGenericOp(const T* srcPtr, T* dstPtr, T* weightPtr, T* newWeightPtr, T* firstMomentPtr, T* secondMomentPtr, 
                              T alpha, T beta1, T beta2, const int epoch, int nelem, int buffIdx) {
    uint32_t nbytes = nelem < 0 ? 0 : nelem*sizeof(T);
    uint32_t npack = DIVUP(nbytes, sizeof(uint64_t));
    uint64_t* srcPack = (uint64_t*)srcPtr;
    uint64_t* dstPack = (uint64_t*)dstPtr;
    uint64_t* weightPack = (uint64_t*)weightPtr;
    uint64_t* newWeightPack = (uint64_t*)newWeightPtr;
    uint64_t* firstMomentPack = (uint64_t*)firstMomentPtr;
    uint64_t* secondMomentPack = (uint64_t*)secondMomentPtr;

    int offset = tid;

    // Always waitSend in case of cleanup
    if (SEND_UPDATE || SEND) waitSend(npack*sizeof(union ncclLLFifoLine));

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
      if (!SEND_UPDATE && SEND) {
        for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, val, sendFlag(i));
        storeLL(sendPtr(0)+offset, val, sendFlag(0));
      }

      if (!SEND_UPDATE && DST) {
        if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {
          // Last incomplete word
          storeAL(dstPack+offset, val, nbytes & 0x7);
        } else {
          storeAL(dstPack+offset, val, sizeof(uint64_t));
        }
      }
      
      static_assert((SEND_UPDATE == 1) ? WEIGHT_UPDATE == 1 : true, "SEND_UPDATE only with WEIGHT_UPDATE");

      if (WEIGHT_UPDATE) {
        //TODO: Following code assumes T is float and hence, will not work when T = i8, i16
        if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {
          // Last incomplete word
          assert(false);
          //TODO: Not handling this yet.
        } else if (firstMomentPack != NULL && secondMomentPack != NULL) {
          const uint64_t old_m = *(firstMomentPack+offset);
          const uint64_t old_v = *(secondMomentPack+offset);

          //m[t] = beta1 * m[t-1] + (1-beta1)*g
          const uint64_t new_m = MULTI<FuncFirstMomentUpdate<T>, T>()(old_m, val, beta1);
          storeAL(firstMomentPack+offset, new_m, sizeof(uint64_t));

          //v[t] = beta2 * v[t-1] + (1-beta2)*g*g
          const uint64_t new_v = MULTI<FuncSecondMomentUpdate<T>, T>()(old_v, val, beta2);
          storeAL(secondMomentPack+offset, new_v, sizeof(uint64_t));

          //m_[t] = m[t]/(1-beta1^t)
          const uint64_t m_ = MULTI<FuncBiasCorrection<T>, T>()(new_m, beta1, epoch+1);

          //v_[t] = v[t]/(1-beta2^t)
          const uint64_t v_ = MULTI<FuncBiasCorrection<T>, T>()(new_v, beta2, epoch+1);

          //w[t] = w[t-1] - alpha*m_[t]/(sqrt(v_[t]) + epsilon)
          uint64_t update = MULTI<FuncAdamWeightUpdate<T>, T>()(*(weightPack+offset), m_, v_, alpha, 1e-6);

          if(SEND_UPDATE) {
            for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, update, sendFlag(i));
            storeLL(sendPtr(0)+offset, update, sendFlag(0));

            if (DST) {
              if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {
                // Last incomplete word
                storeAL(newWeightPack+offset, update, nbytes & 0x7);
              } else {
                storeAL(newWeightPack+offset, update, sizeof(uint64_t));
              }
            }
          } else {
            *(newWeightPack+offset) = update;
          }
        } else {
          val = MULTI<FuncProd<T>, T>()(val, alpha);
          uint64_t update = MULTI<FuncSum<T>, T>()(*(weightPack+offset), val);
          if(SEND_UPDATE) {
            if(buffIdx == 1) {
             // printf("buffIdx 1 offset %d update \n", offset);
              // if (offset == 0) {
              //   printf("address %p\n", weightPack+offset);
              //   MULTI<FuncPrint<T>, T>()(update);
              // }
            }
            for (int i=1; i<NSEND && i<nsend; i++) storeLL(sendPtr(i)+offset, update, sendFlag(i));
            storeLL(sendPtr(0)+offset, update, sendFlag(0));

            if (DST) {
              if (((offset*sizeof(uint64_t)) ^ nbytes) < sizeof(uint64_t)) {
                // Last incomplete word
                storeAL(newWeightPack+offset, update, nbytes & 0x7);
              } else {
                storeAL(newWeightPack+offset, update, sizeof(uint64_t));
              }
            }
          } else {
            *(newWeightPack+offset) = update;
          }
        }
      }
    }
    FOR_RECV(incRecv); if (RECV) postRecv();
    FOR_SEND(incSend, offset);
  }

  __device__ uint64_t* scattered_ptr(int offset, int nelem, T* nextBuffs[4], size_t nextBuffSizes[4], int nextOffsets[4], int& newOffset)
  {
    int idx;
    newOffset = offset;
    for (idx = 0; idx < 4 && nextBuffs[idx] != nullptr; idx++) {
      if (nextOffsets[idx]/(sizeof(uint64_t)/sizeof(T)) > offset) {
        break;
      }

      size_t sz = (idx == 0) ? nextOffsets[idx]/(sizeof(uint64_t)/sizeof(T)) : (nextOffsets[idx]/(sizeof(uint64_t)/sizeof(T)) - nextOffsets[idx-1]/(sizeof(uint64_t)/sizeof(T)));
      newOffset -= sz;
    }

    idx = idx - 1;
    return (uint64_t*)nextBuffs[idx];
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
  ncclLLPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), opCount(opCount) {
    // Make sure step is updated before we read it.
    barrier();
    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) {loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i);}
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) {loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i);}
    loadRecvSync();
    loadSendSync();
  }

  __device__ void send(const T* src, int nelem) {
    return LLGenericOp<0, 1, 1, 0, 0, 0>(src, NULL, NULL, NULL, NULL, NULL, (T)0.0, (T)0, (T)0, (T)0, nelem, 0);
  }

  __device__ void recv(T* dst, int nelem) {
    return LLGenericOp<1, 0, 0, 1, 0, 0>(NULL, dst, NULL, NULL, NULL, NULL, (T)0.0, (T)0, (T)0, (T)0, nelem, 0);
  }

  __device__ void recvAndWeightUpdate(T* dst, T* weight, T* newWght, int nelem) {
    return LLGenericOp<1, 0, 0, 1, 1, 0>(NULL, dst, weight, newWght, NULL, NULL, (T)0.0, (T)0, (T)0, (T)0, nelem, 0);
  }

  __device__ void recvReduceSend(const T* src, int nelem) {
    return LLGenericOp<1, 1, 1, 0, 0, 0>(src, NULL, NULL, NULL, NULL, NULL, (T)0.0, (T)0, (T)0, (T)0, nelem, 0);
  }

  __device__ void recvReduceCopy(const T* src, T* dst, int nelem) {
    return LLGenericOp<1, 0, 1, 1, 0, 0>(src, dst, NULL, NULL, NULL, NULL, (T)0.0, (T)0, (T)0, (T)0, nelem, 0);
  }

  __device__ void copySend(const T* src, T* dst, int nelem) {
    return LLGenericOp<0, 1, 1, 1, 0, 0>(src, dst, NULL, NULL, NULL, NULL, (T)0.0, (T)0, (T)0, (T)0, nelem, 0);
  }

  __device__ void recvCopySend(T* dst, int nelem) {
    return LLGenericOp<1, 1, 0, 1, 0, 0>(NULL, dst, NULL, NULL, NULL, NULL, (T)0.0, (T)0, (T)0, (T)0, nelem, 0);
  }

  __device__ void recvCopySendWeightUpdate(T* dst, T* wght, T* newWght, int nelem) {
    return LLGenericOp<1, 1, 0, 1, 1, 0>(NULL, dst, wght, newWght, NULL, NULL, (T)0.0, (T)0, (T)0, (T)0, nelem, 0);
  }

  __device__ void recvReduceCopySend(const T* src, T* dst, int nelem) {
    return LLGenericOp<1, 1, 1, 1, 0, 0>(src, dst, NULL, NULL, NULL, NULL, (T)0.0, (T)0, (T)0, (T)0, nelem, 0);
  }

  __device__ void recvReduceUpdateandSendWeight(const T* src, T* dst, T* wght, T* newWght, T alpha, int nelem, int buffIdx) {
    return LLGenericOp<1, 1, 1, 1, 1, 1>(src, dst, wght, newWght, NULL, NULL, alpha, (T)0, (T)0, (T)0, nelem, buffIdx);
  }
  
  __device__ void recvReduceUpdateandSendWeightInAdam(const T* src, T* dst, T* wght, T* newWght, T* firstMoment, T* secondMoment, 
                                                      T alpha, T beta1, T beta2, int epoch, int nelem, int buffIdx) {
    return LLGenericOp<1, 1, 1, 1, 1, 1>(src, dst, wght, newWght, firstMoment, secondMoment, alpha, beta1, beta2, epoch, nelem, buffIdx);
  }

  __device__ __forceinline__ ~ncclLLPrimitives() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};
