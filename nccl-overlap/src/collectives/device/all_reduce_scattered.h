/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include <assert.h>
#include <cooperative_groups.h>

//#include <cooperative_group>

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceScatteredRingKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads-WARP_SIZE;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;


  const T ** __restrict__ thisInput = (const T**)args->ThisScatteredSendBuff;
  T** __restrict__ thisWeight = (T**)args->ThisScatteredWeightBuff;
  const T * __restrict__ thisAlpha = (T*)args->ThisAlpha;
  const T alpha = (thisAlpha == NULL) ? (T)0.0 : *thisAlpha;
  const T beta1 = (args->ThisBeta1 == NULL) ? (T)0.0 : *(T*)args->ThisBeta1;
  const T beta2 = (args->ThisBeta1 == NULL) ? (T)0.0 : *(T*)args->ThisBeta2;
  T* __restrict__ thisFirstMoment = (T*)args->ThisScatteredFirstMoment;
  T* __restrict__ thisSecondMoment = (T*)args->ThisScatteredSecondMoment;
  const int epoch = args->epoch;
  size_t* scatteredBuffSizes = args->ThisScatteredBuffSizes;
  int smallNBuff = args->ThisScatteredSmallNBuff;
  const ssize_t nBuff = args->nbuff;
  OptimizerType optimType = args->optimizerType;
  const size_t* __restrict__ buffIdToParentBufferId = args->buffIdToParentBufferId;
  const size_t* parentBuffSizes = args->parentBuffSizes;
  
  int partNum = 0;
  int maxPartSize = min(chunkSize, DIVUP(size,nranks*args->nChannels));
  ALIGN_SIZE(maxPartSize, nthreads*sizeof(uint64_t)/sizeof(T));

  ncclScatteredPrimitives<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, args->nThreads, &ring->prev, &ring->next, thisWeight, stepSize, channel, comm, args->opCount, scatteredBuffSizes, 
    smallNBuff, size, nranks, args->nChannels, loopSize, maxPartSize, nBuff, buffIdToParentBufferId, parentBuffSizes);

  if (optimType != OptimizerType::LAMB) {
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
      int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.sendScattered(thisInput, offset, nelem);
      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.recvReduceSendScattered(thisInput, offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
      prims.directRecvReduceCopySendAdam(thisInput, thisWeight, thisFirstMoment, thisSecondMoment, 
                                        offset, nelem, alpha, beta1, beta2, epoch, partNum);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.directRecvCopySendScattered(thisWeight, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
      // Final wait/copy.
      prims.directRecvScattered(thisWeight, offset, nelem);

      partNum++;
    }
  } else {
    double* __restrict__ weightNormBuff = ((double*)args->weightNormBuff);
    double* __restrict__ rNormBuff = ((double*)args->weightNormBuff) + nBuff;
    T* __restrict__ rStorageBuff = (T*)args->rStorageBuff;

    for (int o = threadIdx.x + blockDim.x * blockIdx.x; o < nBuff*2; o+= gridDim.x*blockDim.x) {
      weightNormBuff[o] = 0.0;
    }

    cooperative_groups::this_grid().sync();

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
      int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.sendScattered(thisInput, offset, nelem);
      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.recvReduceSendScattered(thisInput, offset, nelem);
      }

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
      //atomicAdd(rNormBuff, nelem);
      prims.directRecvReduceCopyLAMB(thisInput, thisWeight, thisFirstMoment, thisSecondMoment, rStorageBuff,
                                     offset, nelem, alpha, beta1, beta2, epoch, partNum, weightNormBuff,
                                     rNormBuff);
      partNum++;
    }

    cooperative_groups::this_grid().sync();
    
    // if (threadIdx.x + blockIdx.x*blockDim.x == 0) {
    //   printf("213: rank %d weightNorm %lf totalElems %lf\n", comm->rank, (double)weightNormBuff[0], (double)rNormBuff[0]);
    //   // printf("213: weightNorm[1] %lf rNorm[1] %lf\n", (double)weightNormBuff[1], (double)rNormBuff[1]);
    // }

    if (true) {
      const int stepSize = channel->buffSize / (sizeof(double)*NCCL_STEPS);
      const int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
      const ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;

      for (ssize_t gridOffset = 0; gridOffset < nBuff*2; gridOffset += nranks*loopSize) {
        int realChunkSize = min(chunkSize, DIVUP(nBuff*2-gridOffset,nranks*args->nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(double));
        ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

        /////////////// begin AllReduce steps ///////////////
        ssize_t offset;
        int nelem;
        int chunk;

        // step 0: push data to next GPU
        chunk = ring->devUserRanks[nranks-1];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, nBuff*2-offset);

        prims.send(weightNormBuff+offset, nelem);
        // k-2 steps: reduce and copy to next GPU
        for (int j=2; j<nranks; ++j) {
          chunk = ring->devUserRanks[nranks-j];
          offset = chunkOffset + chunk * realChunkSize;
          nelem = min(realChunkSize, nBuff*2-offset);

          prims.recvReduceSend(weightNormBuff+offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data and push to the next GPU
        chunk = ring->devUserRanks[0];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, nBuff*2-offset);
        prims.recvReduceCopySend(weightNormBuff+offset, weightNormBuff+offset, nelem);
        
        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          chunk = ring->devUserRanks[nranks-j];
          offset = chunkOffset + chunk * realChunkSize;
          nelem = min(realChunkSize, nBuff*2-offset);

          prims.recvCopySend(weightNormBuff+offset, nelem);
        }

        // Make final copy from buffer to dest.
        chunk = ring->devUserRanks[1];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, nBuff*2-offset);
        // Final wait/copy.
        prims.recv(weightNormBuff+offset, nelem);
      }
    }

    // if (threadIdx.x + blockIdx.x*blockDim.x == 0) {
    //   for (ssize_t i = 0; i < nBuff; i++) {
    //     printf("rank %d norm %ld %f\n", comm->rank, i, (float)weightNormBuff[i]);
    //   }
    // }
    // if (threadIdx.x + blockIdx.x*blockDim.x == 0) {
    //   printf("213: weightNorm %lf rNorm %lf\n", (double)weightNormBuff2[0], (double)rNormBuff2[0]);
    //   printf("213: weightNorm[1] %lf rNorm[1] %lf\n", (double)weightNormBuff2[1], (double)rNormBuff2[1]);
    // }

    cooperative_groups::this_grid().sync();

    for (ssize_t o = threadIdx.x + blockDim.x * blockIdx.x; o < nBuff; o += gridDim.x*blockDim.x) {
      weightNormBuff[o] = sqrt(weightNormBuff[o]);
    }

    for (int o = threadIdx.x + blockDim.x * blockIdx.x; o < nBuff; o += gridDim.x*blockDim.x) {
      rNormBuff[o] = sqrt(rNormBuff[o]);
    }

    cooperative_groups::this_grid().sync();

    // if (threadIdx.x + blockIdx.x*blockDim.x == 0 && comm->rank == 0) {
    //   for (int i = 0; i < nBuff; i++)
    //     printf("weightNorm[%d] %lf rNorm[%d] %lf\n", i, weightNormBuff[i], i, rNormBuff[i]);
    //   // printf("223: weightNorm[1] %lf rNorm[1] %lf\n", (double)weightNormBuff[1], (double)rNormBuff[1]);
    // }

    partNum = 0;

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
      int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
      ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
      ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);

      prims.directSendScatteredLAMB((const T**)thisWeight, rStorageBuff, offset, nelem, partNum, alpha, weightNormBuff, rNormBuff);

      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = chunkOffset + chunk * realChunkSize;
        nelem = min(realChunkSize, size-offset);

        prims.directRecvCopySendScattered(thisWeight, offset, nelem);
      }

      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
      // Final wait/copy.
      prims.directRecvScattered(thisWeight, offset, nelem);

      partNum++;
    }
  }
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceScatteredTreeKernel(struct CollectiveArgs* args) {
   if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("Here2");
 const int tid = threadIdx.x;
  const int nthreads = args->nThreads-WARP_SIZE;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const ssize_t size = args->N;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  int chunkSize = args->lastChunkSize;
  const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;
  
  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclPrimitives<UNROLL/2, 1, 1, T, NCCL_MAX_TREE_ARITY, 1, FUNC> prims(tid, args->nThreads, tree->down, &tree->up, NULL, stepSize, channel, comm, args->opCount, nullptr,0,0);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.send(thisInput+offset, nelem);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclPrimitives<UNROLL/2, 1, 1, T, 1, NCCL_MAX_TREE_ARITY, FUNC> prims(tid, args->nThreads, &tree->up, tree->down, NULL, stepSize, channel, comm, args->opCount, nullptr,0,0);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.recv(thisOutput+offset, nelem);
      } else {
        prims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceScatteredCollNetKernel(struct CollectiveArgs* args) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("Here3");
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads-WARP_SIZE;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const ssize_t size = args->N;
  const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  int chunkSize = args->lastChunkSize;
  const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  if (blockIdx.x < args->nChannels) { // first half of the channels do reduce
    struct ncclTree* tree = &channel->collTreeUp;
    ncclPrimitives<UNROLL, 1, 1, T, 1, 1, FUNC> prims(tid, args->nThreads, tree->down, &tree->up, NULL, stepSize, channel, comm, args->opCount, nullptr,0,0);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.send(thisInput+offset, nelem);
      } else {
        prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  }

  if (blockIdx.x >= args->nChannels) { // second half of the channels do broadcast
    struct ncclTree* tree = &channel->collTreeDn;
    ncclPrimitives<UNROLL, 1, 1, T, 1, 1, FUNC> prims(tid, args->nThreads, &tree->up, tree->down, NULL, stepSize, channel, comm, args->opCount, nullptr, 0,0);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.recv(thisOutput+offset, nelem);
      } else {
        prims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  }
}

template<typename T>
__device__ T* get_buff_for_offset_(T ** __restrict__ thisScatteredPtrs, const size_t* buffSizes, 
                                  const size_t nBuff, ssize_t offset, int nelem, 
                                  T* nextBuffs[4], size_t nextBuffSizes[4], int nextOffsets[4], int& buffIdx) {
  //TODO: Following code assumes that, all input buffers (and output buffers) are contiguous
  //and has equal size, which is a multiple of chunkSize.
  if(nBuff <= 0) {
    printf("%s:%d nBuff %ld\n", __FILE__, __LINE__, nBuff);
  }

  assert(nBuff > 0);
  buffIdx = nBuff - 1;
  T* buff_ptr = nullptr;
  size_t old_offset = offset;
  size_t buffOffset = 0;
  for (size_t i = 0; i < nBuff; i++) {
    if (offset < buffSizes[i]) {
      buffIdx = i;
      buff_ptr = thisScatteredPtrs[i] + offset;
      buffOffset = offset;
      break;
    }

    offset -= buffSizes[i];
  }

  offset = 0;
  // if(threadIdx.x + blockDim.x * blockIdx.x == 0) {
  //   printf("old_offset %ld nelem %d nextOffsets[0] %d nextOffsets[1] %d\n", old_offset, nelem, nextOffsets[0], nextOffsets[1]);
//  }
  for (size_t i = buffIdx, j = 0; i < nBuff && j < nBuff; i++, j++) {
    // if(threadIdx.x + blockDim.x * blockIdx.x == 0) {
    //   printf("old_offset %ld nelem %d nextOffsets[0] %d nextOffsets[1] %d i %ld j %ld\n", old_offset, nelem, nextOffsets[0], nextOffsets[1], i, j);
    // }
    if (nelem > 0) {
      nextOffsets[j] = offset;
      if (i == buffIdx)
        nextBuffs[j] = buff_ptr;
      else
        nextBuffs[j] = thisScatteredPtrs[i];
      
      nextBuffSizes[j] = buffSizes[i];
    } else {
      break;
    }
    if (i == buffIdx) {
      nelem -= buffSizes[i] - buffOffset;
      offset += buffSizes[i] - buffOffset;
    }
    else {
      nelem -= buffSizes[i];
      offset += buffSizes[i];
    }
    
  }

  // if(threadIdx.x + blockDim.x * blockIdx.x == 0) {
  //   printf("old_offset %ld nelem %d nextOffsets[0] %d nextOffsets[1] %d\n", old_offset, nelem, nextOffsets[0], nextOffsets[1]);
  // }

  if(buffIdx == 0 && offset > 8192)
    printf("Invalid offset %ld\n", old_offset);
  
  return buff_ptr;
}

template<typename T>
__device__ T* get_buff_for_offset(const T ** __restrict__ thisScatteredPtrs, const size_t* buffSizes, 
                                  const size_t nBuff, ssize_t offset, int nelem, 
                                  T* nextBuffs[4], size_t nextBuffSizes[4], int nextOffsets[4], int& buffIdx) {
  return get_buff_for_offset_((T**) thisScatteredPtrs, buffSizes, nBuff, offset, nelem, nextBuffs, nextBuffSizes, nextOffsets, buffIdx);
}

//template<typename T>
// void memset(void* buff, size_t v, int nelem) {
//   for (int i = 0 ; i < nelem; i++) {
//     buff[i] = v;
//   }
// }

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceScatteredRingLLKernel(struct CollectiveArgs* args) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("Here4");
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;

  ncclLLScatteredPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount);

  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);

  const ssize_t loopSize = args->nChannels*nranks*chunkSize;
  // Compute pointers
  const T ** __restrict__ thisScatteredInput = (const T**)args->ThisScatteredSendBuff;
  T ** __restrict__ thisScatteredWeight = (T**)args->ThisScatteredWeightBuff;
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  const T * __restrict__ thisAlpha = (T*)args->ThisAlpha;
  const T alpha = (thisAlpha == NULL) ? (T)0.0 : *thisAlpha;

  if (thisInput != nullptr and thisScatteredWeight != nullptr and thisScatteredInput == nullptr) {
    //Grads are contiguous but weights are scattered
    assert(false);
    return;
    const T alpha = 1;
    // if (threadIdx.x + blockIdx.x*blockDim.x == 0)
    //   printf ("thisWeight %p thisNewWeight %p\n", thisWeight, thisNewWeight);

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      chunkSize = min(DIVUP(size-gridOffset, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);
      T* nextBuffs[4] = {nullptr};
      int nextOffsets[4] = {0};
      size_t nextBuffSizes[4] = {0};
      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);
      // if (!(args->ThisScatteredBuffSizes[0]%nelem == 0 and args->ThisScatteredBuffSizes[0]%chunkSize == 0)) {
      //   printf("dsfsdfdsf\n");
      // }
      //assert(args->ThisScatteredBuffSizes[0]%nelem == 0 and args->ThisScatteredBuffSizes[0]%chunkSize == 0);
      int buffIdx = 0;
      const T* ptr = thisInput + offset;//get_buff_for_offset((T**)thisScatteredInput, args->ThisScatteredBuffSizes, 
                     //                    args->ThisScatteredNBuff, offset, nelem, nextBuffs, nextBuffSizes,
                     //                    nextOffsets, buffIdx);
      // if(buffIdx == 1) {
      //   printf("nelem %d offset %ld chunk %d chunkSize %ld gridOffset %ld size %ld loopSize %ld\n", nelem, offset, chunk, chunkSize, gridOffset, size, loopSize);
      // }

      if(buffIdx==1) {
        // printf("offset %ld nelem %d nextBuffs[0] %d nextBuffs[1] %d nextBuffs[2] %d nextBuffs[3] %d nextOffsets[0] %d nextOffsets[1] %d nextOffsets[2] %d nextOffsets[3] %d\n", 
        // offset, nelem, nextBuffs[0], nextBuffs[1], nextBuffs[2], nextBuffs[3], nextOffsets[0], nextOffsets[1], nextOffsets[2], nextOffsets[3]);
      }
      LLprims.sendScatteredWeights(ptr, nextBuffs, nextBuffSizes, nextOffsets, nelem, thisScatteredWeight, buffIdx);
      
      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        // LLprims.recvReduceSendScattered(get_buff_for_offset((T**)thisScatteredInput, args->ThisScatteredBuffSizes, 
        //                                            args->ThisScatteredNBuff, offset, nelem, nextBuffs, nextBuffSizes,
        //                                            nextOffsets, buffIdx), nextBuffs, nextBuffSizes, nextOffsets, nelem, thisScatteredWeight, buffIdx);
        LLprims.recvReduceSendScatteredWeights(thisInput+offset, nextBuffs, nextBuffSizes, nextOffsets, nelem, thisScatteredWeight, buffIdx);
      }
      
      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      if (false) {// && thisWeight == nullptr) {
        //LLprims.recvReduceCopySend(thisInput, thisOutput+offset, nelem);
      } else if (false){// && doAdam) {
        // LLprims.recvReduceUpdateandSendWeightInAdam(thisInput+offset, thisOutput+offset, thisWeight+offset, thisWeight+offset, 
        //                                             thisFirstMoment+offset, thisSecondMoment+offset, alpha, beta1, beta2, epoch, nelem);
      } else {
        //SGD
        //Reduce the gradients, then update and send weights
        const T* input = thisInput+offset;//get_buff_for_offset((T**)thisScatteredInput, args->ThisScatteredBuffSizes, 
                         //                    args->ThisScatteredNBuff, offset, nelem, nextBuffs, 
                         //                    nextBuffSizes, nextOffsets, buffIdx);
        T* weight_offset = get_buff_for_offset_(thisScatteredWeight, args->ThisScatteredBuffSizes, args->nbuff, 
                                               offset, nelem, nextBuffs, 
                                               nextBuffSizes, nextOffsets, buffIdx);
        LLprims.recvReduceUpdateandSendWeightScatteredWeights(input, nullptr, weight_offset, alpha, nextBuffs, nextBuffSizes,
                                                       nextOffsets, nelem, thisScatteredWeight, buffIdx);
      }
      
      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        if (false) {// && thisWeight == nullptr) {
          //LLprims.recvCopySend(thisOutput+offset, nelem);
        } else {
          //Receive and send the updated weights
          //Same for SGD and Adam
          LLprims.recvCopySendScatteredWeights(get_buff_for_offset_(thisScatteredWeight, args->ThisScatteredBuffSizes, 
                                                   args->nbuff, offset, nelem, nextBuffs, 
                                                   nextBuffSizes, nextOffsets, buffIdx), nextBuffs, nextBuffSizes, nextOffsets, nelem, thisScatteredWeight, buffIdx);
        }
      }
      
      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      // Here we need to copy from buffer to this output.
      if (false){// && thisWeight == nullptr) {
        //LLprims.recv(thisOutput+offset, nelem);
      } else {
        //Receive the updated weights
        //Same for SGD and Adam
        LLprims.recvScatteredWeights(get_buff_for_offset_(thisScatteredWeight, args->ThisScatteredBuffSizes, 
                                         args->nbuff, offset, nelem, nextBuffs, 
                                                   nextBuffSizes, nextOffsets, buffIdx), nextBuffs, nextBuffSizes, nextOffsets, nelem, thisScatteredWeight, buffIdx);
      }
    }
  } else if (thisInput == nullptr and thisScatteredWeight != nullptr and thisScatteredInput != nullptr) {
    //Both grads and weights are scattered

    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      chunkSize = min(DIVUP(size-gridOffset, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);
      /////////////// begin AllReduce steps ///////////////
      ssize_t offset;
      int nelem;
      int chunk;

      // step 0: push data to next GPU
      chunk = ring->devUserRanks[nranks-1];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);
      
      //Get correct offset in the scattered buffers based on offset
      LLprims.sendScatteredGradWeights(thisScatteredInput, alpha, args->ThisScatteredBuffSizes, args->nbuff, offset, nelem);
      
      // k-2 steps: reduce and copy to next GPU
      for (int j=2; j<nranks; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        LLprims.recvReduceSendScatteredGradWeights(thisScatteredInput, alpha, args->ThisScatteredBuffSizes, args->nbuff, offset, nelem);
      }
      
      // step k-1: reduce this buffer and data, which will produce the final
      // result that we store in this data and push to the next GPU
      chunk = ring->devUserRanks[0];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);
      
      if (false) {// && thisWeight == nullptr) {
        //LLprims.recvReduceCopySend(thisInput, thisOutput+offset, nelem);
      } else if (false){// && doAdam) {
        // LLprims.recvReduceUpdateandSendWeightInAdam(thisInput+offset, thisOutput+offset, thisWeight+offset, thisWeight+offset, 
        //                                             thisFirstMoment+offset, thisSecondMoment+offset, alpha, beta1, beta2, epoch, nelem);
      } else {
        //SGD
        //Reduce the gradients, then update and send weights
        LLprims.recvReduceUpdateandSendScatteredGradWeights(thisScatteredInput, thisScatteredWeight, alpha, args->ThisScatteredBuffSizes, 
                                                             args->nbuff, offset, nelem);
      }
      
      
      // k-2 steps: copy to next GPU
      for (int j=1; j<nranks-1; ++j) {
        chunk = ring->devUserRanks[nranks-j];
        offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
        nelem = min(chunkSize, size-offset);

        if (false) {// && thisWeight == nullptr) {
          //LLprims.recvCopySend(thisOutput+offset, nelem);
        } else {
          //Receive and send the updated weights
          //Same for SGD and Adam
          LLprims.recvCopySendScatteredGradWeights(thisScatteredWeight, args->ThisScatteredBuffSizes, 
                                                     args->nbuff, offset, nelem);
        }
      }
      
      // Make final copy from buffer to dest.
      chunk = ring->devUserRanks[1];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      // Here we need to copy from buffer to this output.
      if (false) {// && thisWeight == nullptr) {
        //LLprims.recv(thisOutput+offset, nelem);
      } else {
        //Receive the updated weights
        //Same for SGD and Adam
        LLprims.recvScatteredGradWeights(thisScatteredWeight, args->ThisScatteredBuffSizes, 
                                         args->nbuff, offset, nelem);
      }
    }
  } else {
    if(threadIdx.x + blockDim.x*blockIdx.x == 0)
      printf("Invalid type: thisInput '%p', thisScatteredWeight '%p', thisScatteredInput '%p'\n", thisInput, thisScatteredWeight, thisScatteredInput);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceScatteredTreeLLKernel(struct CollectiveArgs* args) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("Here5");
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const ssize_t size = args->N;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclLLPrimitives<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreads, tree->down, &tree->up, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclLLPrimitives<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, &tree->up, tree->down, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceScatteredCollNetLLKernel(struct CollectiveArgs* args) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("Here6");
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  const ssize_t size = args->N;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
  const ssize_t loopSize = args->nChannels*chunkSize;
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  if (blockIdx.x < args->nChannels) { // first half of the channels do reduce
    struct ncclTree* tree = &channel->collTreeUp;
    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, tree->down, &tree->up, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  }

  if (blockIdx.x >= args->nChannels) { // second half of the channels do broadcast
    struct ncclTree* tree = &channel->collTreeDn;
    ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &tree->up, tree->down, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  }
}

#include "prims_ll128.h"
template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceScatteredRingLL128Kernel(struct CollectiveArgs* args) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("Here7");
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  
  ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount);

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = (NCCL_LL128_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));
  // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;

  const ssize_t loopSize = args->nChannels*nranks*chunkSize;

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(thisOutput+offset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceScatteredTreeLL128Kernel(struct CollectiveArgs* args) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    printf("Here8");
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclTree* treeUp = &channel->treeUp;
  struct ncclTree* treeDn = &channel->treeDn;
  const ssize_t size = args->N;
  ssize_t chunkSize = args->lastChunkSize;
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/8;
  const ssize_t loopSize = args->nChannels*chunkSize;
  int nthreadsSplit = NCCL_LL128_SPLIT(nthreads);
  
  if (loopSize > size) {
    chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  }

  // Compute pointers
  const T * __restrict__ thisInput = (const T*)args->ThisInput;
  T * __restrict__ thisOutput = (T*)args->ThisOutput;

  if (treeUp->up == -1) {
    // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
    ncclLL128Primitives<T, FUNC, NCCL_MAX_TREE_ARITY, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, treeUp->down, treeDn->down, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
    }
  } else {
    if (tid < nthreadsSplit) {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclLL128Primitives<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreadsSplit, treeUp->down, &treeUp->up, channel, comm, args->opCount);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeUp->down[0] == -1) {
          LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    } else {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclLL128Primitives<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid-nthreadsSplit, nthreads-nthreadsSplit, &treeDn->up, treeDn->down, channel, comm, args->opCount);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeDn->down[0] == -1) {
          LLprims.recv(thisOutput+offset, nelem);
        } else {
          LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceScatteredCollNetLL128Kernel(struct CollectiveArgs* args) { }
