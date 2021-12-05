/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives_computation.h"
#include "collectives.h"
#include <cooperative_groups.h>

#define TYPE_ALL_REDUCE 1








#if TYPE_ALL_REDUCE == 0

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceComputationRingKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int nthreads = args->nThreads-WARP_SIZE;
  const int bid = args->bid;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->N;
  const int nranks = comm->nRanks;
  // assert(channel->buffSize % 3 == 0);//Divide channel into two parts of 1/3 and 2/3
  int stepSize = (channel->buffSize) / (sizeof(T)*NCCL_STEPS);
  int chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  ssize_t loopSize = args->nChannels*(ssize_t)chunkSize;

  // Compute pointers
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;
  
  float lr;
  lr = (float)args->lr;
  float beta1;
  beta1 = (float)args->beta1;
  float beta2;
  beta2 = (float)args->beta2;
  const half* __restrict__ g;
  g = (const half*)args->g;
  half* __restrict__ halfw;
  halfw = (half*)args->halfw;
  float* __restrict__ w;
  w = (float*)args->w;
  float* __restrict__ m;
  m = (float*)args->m;
  float* __restrict__ v;
  v = (float*)args->v;

  /*RingSimple: {INSERT SHARED MEMORY FOR REDUCTION}*/
  ncclPrimitivesComputation<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, 1, 1, half, FuncSum<half>, float, FuncSum<float>>
    prims(tid, args->nThreads, &ring->prev, &ring->next, halfw, stepSize, channel, comm, args->opCount);

  
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
    prims.send(g+ offset, nelem);
    //prims.send(thisInput+offset, nelem);
    // if (threadIdx.x == 0) {
    //   printf("67: offset %ld\n", offset);
    // }
    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
      prims.recvReduceSend(g+ offset, nelem);
      //prims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);
    
    prims.recvReduceCopySend(g + offset, halfw + offset, w + offset, lr, beta1, beta2, m + offset, v + offset, nelem);
    //prims.recvReduceCopySendTF16(g + offset, halfw + offset, 0, 0, 0, nullptr, nullptr, 0,nelem);
  // }
  
  // for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
  //   int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
  //   ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
  //   ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

  //   /////////////// begin AllReduce steps ///////////////
  //   ssize_t offset;
  //   int nelem;
  //   int chunk;
  //   chunk = ring->devUserRanks[0];
  //   offset = chunkOffset + chunk * realChunkSize;
  //   nelem = min(realChunkSize, size-offset);
    
  //   prims.send(halfw+offset, nelem);

    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
      prims.recvCopySend(halfw + offset, nelem);
      //prims.directRecvCopySend(thisOutput+offset, offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    // Final wait/copy.
    prims.recv(halfw + offset, nelem);
    // if (threadIdx.x== 0) {
    //   printf("offset %ld\n", offset);
    // }
    // if (threadIdx.x == 0)
    //   printf("83: nelem %ld\n", nelem);
  }
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceComputationTreeKernel(struct CollectiveArgs* args) {
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
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;
    T lr;
  lr = (T)args->lr;
  T beta1;
  beta1 = (T)args->beta1;
  T beta2;
  beta2 = (T)args->beta2;
  const T* __restrict__ g;
  g = (const T*)args->g;
  T* __restrict__ w;
  w = (T*)args->w;
  T* __restrict__ m;
  m = (T*)args->m;
  T* __restrict__ v;
  v = (T*)args->v;

#if 0

  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclPrimitivesComputation<UNROLL/2, 1, 1, T, NCCL_MAX_TREE_ARITY, 1, FUNC> prims(tid, args->nThreads, tree->down, &tree->up, NULL, stepSize, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.recvReduceCopy(g + offset, (T*)(g + offset), nelem);

    if (nelem > 0) {
      const size_t packFactor = 4;
      uint2* gPack = (uint2*)(g+ offset);
      ulong2* wPack = (ulong2*)((float*)w + offset);
      ulong2* mPack = (ulong2*)((float*)m + offset);
      ulong2* vPack = (ulong2*)((float*)v + offset);
      for (size_t ii = threadIdx.x; ii < nelem/packFactor; ii += blockDim.x) {
        ulong2 S2;
        MULTI128<mixedbinOp1<T>, T>().mixedbinOp1(beta2, *(vPack + ii), *(gPack + ii), S2);
        *(vPack + ii) = S2;
        ulong2 S4;
        MULTI128<binOp2<T>, T>().binOp2(beta2, S2, S4);
        ulong2 S1;
        MULTI128<mixedbinOp3<T>, T>().mixedbinOp3(beta1, *(mPack + ii), *(gPack + ii), S1);
        *(mPack + ii) = S1;
        ulong2 S3;
        MULTI128<binOp4<T>, T>().binOp4(beta1, S1, S3);
        ulong2 S5;
        MULTI128<binOp5<T>, T>().binOp5(lr, *(wPack + ii), S3, S4, S5);
        *(wPack + ii) = S5;
      }
    __syncthreads();
    }

        // prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.send(g+ offset, nelem);
        // prims.send(thisInput+offset, nelem);
      } else {
        prims.recvReduceSend(g+ offset, nelem);
        // prims.recvReduceSend(thisInput+offset, nelem);
      }
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclPrimitivesComputation<UNROLL/2, 1, 1, T, 1, NCCL_MAX_TREE_ARITY, FUNC> prims(tid, args->nThreads, &tree->up, tree->down, NULL, stepSize*2, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        prims.send((T*)((float*)w + offset), nelem * 2);

        // prims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        prims.recv((T*)((float*)w + offset), nelem * 2);

        // prims.recv(thisOutput+offset, nelem);
      } else {
        prims.recvCopySend((T*)((float*)w + offset), nelem * 2);

        // prims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);

#endif
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceComputationCollNetKernel(struct CollectiveArgs* args) {
  // const int tid = threadIdx.x;
  // const int nthreads = args->nThreads-WARP_SIZE;
  // const int bid = args->bid;
  // struct ncclDevComm* comm = args->comm;
  // struct ncclChannel* channel = comm->channels+blockIdx.x;
  // const ssize_t size = args->N;
  // const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  // int chunkSize = args->lastChunkSize;
  // const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  // const ssize_t loopSize = args->nChannels*chunkSize;

  // if (loopSize > size) {
  //   chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  // }

  // // Compute pointers
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;
  assert(false);
  // if (blockIdx.x < args->nChannels) { // first half of the channels do reduce
  //   struct ncclTree* tree = &channel->collTreeUp;
  //   ncclPrimitivesComputation<UNROLL, 1, 1, T, 1, 1, FUNC> prims(tid, args->nThreads, tree->down, &tree->up, NULL, stepSize, channel, comm, args->opCount);
  //   for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  //     // Up
  //     ssize_t offset = gridOffset + bid*chunkSize;
  //     int nelem = min(chunkSize, size-offset);
  //     if (tree->up == -1) {
  //       prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
  //     } else if (tree->down[0] == -1) {
  //       prims.send(thisInput+offset, nelem);
  //     } else {
  //       prims.recvReduceSend(thisInput+offset, nelem);
  //     }
  //   }
  // }

  // if (blockIdx.x >= args->nChannels) { // second half of the channels do broadcast
  //   struct ncclTree* tree = &channel->collTreeDn;
  //   ncclPrimitivesComputation<UNROLL, 1, 1, T, 1, 1, FUNC> prims(tid, args->nThreads, &tree->up, tree->down, NULL, stepSize, channel, comm, args->opCount);
  //   for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  //     // Down
  //     ssize_t offset = gridOffset + bid*chunkSize;
  //     int nelem = min(chunkSize, size-offset);
  //     if (tree->up == -1) {
  //       prims.send(thisOutput+offset, nelem);
  //     } else if (tree->down[0] == -1) {
  //       prims.recv(thisOutput+offset, nelem);
  //     } else {
  //       prims.recvCopySend(thisOutput+offset, nelem);
  //     }
  //   }
  // }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationRingLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;

  ncclLLPrimitivesComputation<half, FUNC, float, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount);

  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);

  const ssize_t loopSize = args->nChannels*nranks*chunkSize;

  // Compute pointers
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;
    float lr;
  lr = (float)args->lr;
  float beta1;
  beta1 = (float)args->beta1;
  float beta2;
  beta2 = (float)args->beta2;
  const half* __restrict__ g;
  g = (const half*)args->g;
  half* __restrict__ halfw;
  halfw = (half*)args->halfw;
  float* __restrict__ w;
  w = (float*)args->w;
  float* __restrict__ m;
  m = (float*)args->m;
  float* __restrict__ v;
  v = (float*)args->v;

  /*RINGLL: {INSERT SHARED MEMORY FOR REDUCTION}*/
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

    LLprims.send(g+ offset, nelem);
    //LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(g+ offset, nelem);
      //LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(lr, beta1, beta2, g + offset, halfw + offset, w + offset, m+offset, v+offset , nelem);

    //LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
/*RINGLL: REDUCTION {END FOR LOOP FOR}*/
    
/*RINGLL: REDUCTION {PER-GPU REDUCTION}*/

/*RINGLL: REDUCTION {TRANSFER}*/
    
  /*RINGLL: REDUCTION {BEGIN FOR LOOP FOR}*/
  /*RINGLL: REDUCTION {COMPUTATION}*/
    // k-2 steps: copy to next GPU
  
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);
      LLprims.recvCopySend((halfw + offset),  nelem);
      //LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv((halfw + offset),  nelem);
    // //LLprims.recv(thisOutput+offset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationTreeLLKernel(struct CollectiveArgs* args) {
#if 0
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
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;
    T lr;
  lr = (T)args->lr;
  T beta1;
  beta1 = (T)args->beta1;
  T beta2;
  beta2 = (T)args->beta2;
  const T* __restrict__ g;
  g = (const T*)args->g;
  T* __restrict__ w;
  w = (T*)args->w;
  T* __restrict__ m;
  m = (T*)args->m;
  T* __restrict__ v;
  v = (T*)args->v;

  /*TREELL: {REDUCTION SHMEM}*/
  
  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclLLPrimitivesComputation<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreads, tree->down, &tree->up, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy(g + offset, (T*)(g + offset), nelem);

    if (nelem > 0) {
      const size_t packFactor = 4;
      uint2* gPack = (uint2*)(g+ offset);
      ulong2* wPack = (ulong2*)((float*)w + offset);
      ulong2* mPack = (ulong2*)((float*)m + offset);
      ulong2* vPack = (ulong2*)((float*)v + offset);
      for (size_t ii = threadIdx.x; ii < nelem/packFactor; ii += blockDim.x) {
        ulong2 S2;
        MULTI128<mixedbinOp1<T>, T>().mixedbinOp1(beta2, *(vPack + ii), *(gPack + ii), S2);
        *(vPack + ii) = S2;
        ulong2 S4;
        MULTI128<binOp2<T>, T>().binOp2(beta2, S2, S4);
        ulong2 S1;
        MULTI128<mixedbinOp3<T>, T>().mixedbinOp3(beta1, *(mPack + ii), *(gPack + ii), S1);
        *(mPack + ii) = S1;
        ulong2 S3;
        MULTI128<binOp4<T>, T>().binOp4(beta1, S1, S3);
        ulong2 S5;
        MULTI128<binOp5<T>, T>().binOp5(lr, *(wPack + ii), S3, S4, S5);
        *(wPack + ii) = S5;
      }
    __syncthreads();
    }

        //LLprims.recvReduceCopy2(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(g+ offset, nelem);
        //LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(g+ offset, nelem);
        //LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }

    if (tree->up == -1) {
      /*TREELL: REDUCTION {REDUCTION COMPUTATION}*/

      /*TREELL: REDUCTION {GLOBAL MEM REDUCTION}*/
      /*TREELL: REDUCTION {COMPUTATION}*/
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclLLPrimitivesComputation<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, &tree->up, tree->down, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send((T*)((float*)w + offset), nelem * 2);

        //LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv((T*)((float*)w + offset), nelem * 2);

        //LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend((T*)((float*)w + offset), nelem * 2);

        //LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);
#endif
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationCollNetLLKernel(struct CollectiveArgs* args) {
  assert(false);
  // const int tid = threadIdx.x;
  // const int nthreads = args->nThreads;
  // const int bid = args->bid;
  // struct ncclDevComm* comm = args->comm;
  // struct ncclChannel* channel = comm->channels+blockIdx.x;
  // const ssize_t size = args->N;
  // ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  // const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
  // const ssize_t loopSize = args->nChannels*chunkSize;

  // if (loopSize > size) {
  //   chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  // }

  // // Compute pointers
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;

  // if (blockIdx.x < args->nChannels) { // first half of the channels do reduce
  //   struct ncclTree* tree = &channel->collTreeUp;
  //   ncclLLPrimitivesComputation<T, FUNC, 1, 1> LLprims(tid, nthreads, tree->down, &tree->up, channel, comm, args->opCount);
  //   for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  //     // Up
  //     ssize_t offset = gridOffset + bid*chunkSize;
  //     int nelem = min(chunkSize, size-offset);
  //     if (tree->up == -1) {
  //       LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
  //     } else if (tree->down[0] == -1) {
  //       LLprims.send(thisInput+offset, nelem);
  //     } else {
  //       LLprims.recvReduceSend(thisInput+offset, nelem);
  //     }
  //   }
  // }

  // if (blockIdx.x >= args->nChannels) { // second half of the channels do broadcast
  //   struct ncclTree* tree = &channel->collTreeDn;
  //   ncclLLPrimitivesComputation<T, FUNC, 1, 1> LLprims(tid, nthreads, &tree->up, tree->down, channel, comm, args->opCount);
  //   for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  //     // Down
  //     ssize_t offset = gridOffset + bid*chunkSize;
  //     int nelem = min(chunkSize, size-offset);
  //     if (tree->up == -1) {
  //       LLprims.send(thisOutput+offset, nelem);
  //     } else if (tree->down[0] == -1) {
  //       LLprims.recv(thisOutput+offset, nelem);
  //     } else {
  //       LLprims.recvCopySend(thisOutput+offset, nelem);
  //     }
  //   }
  // }
}

#include "prims_ll128_computation.h"
template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationRingLL128Kernel(struct CollectiveArgs* args) {

  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->N;

  ncclLL128PrimitivesComputation<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount, size);

  
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;
  ssize_t chunkSize = (NCCL_LL128_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));
  // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;

  const ssize_t loopSize = args->nChannels*nranks*chunkSize;

  // Compute pointers
  float lr;
  lr = (float)args->lr;
  float beta1;
  beta1 = (float)args->beta1;
  float beta2;
  beta2 = (float)args->beta2;
  const T* __restrict__ g;
  g = (const T*)args->g;
  T* __restrict__ halfw = (T*)args->halfw;
  float* __restrict__ w;
  w = (float*)args->w;
  float* __restrict__ m;
  m = (float*)args->m;
  float* __restrict__ v;
  v = (float*)args->v;


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
    LLprims.send(g+ offset, nelem);
    //LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);
      LLprims.recvReduceSend(g+ offset, nelem);
      //LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.recvReduceCopySend(lr, beta1, beta2, g+offset, halfw+offset, w+offset, m+offset, v+offset, offset, nelem);
    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);
      LLprims.recvCopySend(halfw + offset,  nelem);
      //LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(halfw + offset,  nelem);
    //LLprims.recv(thisOutput+offset, nelem);
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationTreeLL128Kernel(struct CollectiveArgs* args) {
#if 0
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
    T lr;
  lr = (T)args->lr;
  T beta1;
  beta1 = (T)args->beta1;
  T beta2;
  beta2 = (T)args->beta2;
  const T* __restrict__ g;
  g = (const T*)args->g;
  T* __restrict__ w;
  w = (T*)args->w;
  T* __restrict__ m;
  m = (T*)args->m;
  T* __restrict__ v;
  v = (T*)args->v;

  if (treeUp->up == -1) {
    // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
    ncclLL128PrimitivesComputation<T, FUNC, NCCL_MAX_TREE_ARITY, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, treeUp->down, treeDn->down, channel, comm, args->opCount);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      LLprims.recvReduceCopy(g + offset, (T*)(g + offset),  nelem);
    if (nelem > 0) {
      const size_t packFactor = 4;
      uint2* gPack = (uint2*)(g+ offset);
      ulong2* wPack = (ulong2*)((float*)w + offset);
      ulong2* mPack = (ulong2*)((float*)m + offset);
      ulong2* vPack = (ulong2*)((float*)v + offset);
      for (size_t ii = threadIdx.x; ii < nelem/packFactor; ii += blockDim.x) {
        ulong2 S2;
        MULTI128<mixedbinOp1<T>, T>().mixedbinOp1(beta2, *(vPack + ii), *(gPack + ii), S2);
        *(vPack + ii) = S2;
        ulong2 S4;
        MULTI128<binOp2<T>, T>().binOp2(beta2, S2, S4);
        ulong2 S1;
        MULTI128<mixedbinOp3<T>, T>().mixedbinOp3(beta1, *(mPack + ii), *(gPack + ii), S1);
        *(mPack + ii) = S1;
        ulong2 S3;
        MULTI128<binOp4<T>, T>().binOp4(beta1, S1, S3);
        ulong2 S5;
        MULTI128<binOp5<T>, T>().binOp5(lr, *(wPack + ii), S3, S4, S5);
        *(wPack + ii) = S5;
      }
    __syncthreads();
    }

      for(size_t wOffset = 0; wOffset < nelem * 2;wOffset += nelem) {

      LLprims.send((T*)(((float*)w) + offset) + wOffset, nelem);
    }
      //LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
    }
  } else {
    if (tid < nthreadsSplit) {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclLL128PrimitivesComputation<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreadsSplit, treeUp->down, &treeUp->up, channel, comm, args->opCount);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeUp->down[0] == -1) {
          LLprims.send(g+ offset, nelem);
          // LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(g+ offset, nelem);
          // LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    } else {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclLL128PrimitivesComputation<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid-nthreadsSplit, nthreads-nthreadsSplit, &treeDn->up, treeDn->down, channel, comm, args->opCount);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeDn->down[0] == -1) {
          for(size_t wOffset = 0; wOffset < nelem * 2;wOffset += nelem) {

      LLprims.recv((T*)(((float*)w) + offset) + wOffset,  nelem);
    }
          // LLprims.recv(thisOutput+offset, nelem);
        } else {
          for(size_t wOffset = 0; wOffset < nelem * 2;wOffset += nelem) {

      LLprims.recvCopySend((T*)(((float*)w) + offset) + wOffset,  nelem);
    }
          // LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
#endif
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationCollNetLL128Kernel(struct CollectiveArgs* args) {assert(false); }

#elif TYPE_ALL_REDUCE == 1


template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceComputationRingKernel(struct CollectiveArgs* args) {
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

  // Compute pointers
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;
  float lr;
  lr = (float)args->lr;
  float beta1;
  beta1 = (float)args->beta1;
  float beta2;
  beta2 = (float)args->beta2;
  const T* __restrict__ g;
  g = (const T*)args->g;
  T* __restrict__ halfw;
  halfw = (T*)args->halfw;
  float* __restrict__ r = (float*)args->r;;
  float* __restrict__ w;
  w = (float*)args->w;
  float* __restrict__ m;
  m = (float*)args->m;
  float* __restrict__ v;
  v = (float*)args->v;
  float lambda = (float)1.0f;

  __shared__ float normShMemory[2];

  if (threadIdx.x <= 1) {
    normShMemory[threadIdx.x] = 0.0f;
  }

  __syncthreads();

  /*RingSimple: {INSERT SHARED MEMORY FOR REDUCTION}*/
  int partNum = 0;
  int maxPartSize = min(chunkSize, DIVUP(size,nranks*args->nChannels));
  ALIGN_SIZE(maxPartSize, nthreads*sizeof(uint64_t)/sizeof(T));

  ncclPrimitivesComputation<UNROLL, ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS, T, 1, 1, FUNC>
    prims(tid, args->nThreads, &ring->prev, &ring->next, halfw, stepSize, channel, comm, args->opCount,size,nranks,args->nChannels,loopSize, maxPartSize);

  cooperative_groups::grid_group __grid_group = cooperative_groups::this_grid();

  register float wwVal = 0;
  register float rrVal = 0;

  // for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
  //   int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
  //   ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
  //   ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

  //   /////////////// begin AllReduce steps ///////////////
  //   ssize_t offset;
  //   int nelem;
  //   int chunk;

  //   // step k-1: reduce this buffer and data, which will produce the final
  //   // result that we store in this data and push to the next GPU
  //   chunk = ring->devUserRanks[0];
  //   offset = chunkOffset + chunk * realChunkSize;
  //   nelem = min(realChunkSize, size-offset);

  //   for (int i = threadIdx.x; i < nelem; i += blockDim.x) {
  //     T ww = *(w + offset + i);
  //     wwVal += (float)(ww*ww);
  //   }
  // } 
  

  // if (threadIdx.x == 0) {
  //   printf("51: maxPartSize %d blockIdx.x %d\n", maxPartSize, blockIdx.x);
  // }
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
    prims.send(g+ offset, nelem);
    //prims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
      prims.recvReduceSend(g+ offset, nelem);
      //prims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);
    // if (threadIdx.x == 0) {
    //   printf("loopSize %ld gridOffset %ld\n", loopSize, gridOffset);
    // }
    // if (threadIdx.x == 0) {
    //   printf("84: realChunkSize %d nelem %d partNum %d blockIdx.x %d args->nChannels %d loopSize %ld offset %ld rank %d\n", realChunkSize, nelem, partNum, blockIdx.x, args->nChannels, loopSize, offset, comm->rank);
    // }
    prims.directRecvReduceCopySend(g + offset, halfw+offset, lr, beta1, beta2, w, m, v, r, offset, nelem, partNum, size, args->nChannels, &rrVal, &wwVal);
    partNum++;
  }

  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    wwVal += __shfl_down_sync(0xffffffff, wwVal, offset);
    rrVal += __shfl_down_sync(0xffffffff, rrVal, offset);
  }

  if (threadIdx.x % warpSize == 0)
    ::atomicAdd((float*)(&normShMemory[0]), (float)(wwVal)); 

  if (threadIdx.x % warpSize == 0)
    ::atomicAdd((float*)(&normShMemory[1]), (float)(rrVal)); 
  


  __syncwarp();
  // if (threadIdx.x == 0) {
  //   printf("152: rank %d wwVal %f rrVal %f\n", comm->rank, normShMemory[0], normShMemory[1]);
  // }
  if(threadIdx.x + blockDim.x*blockIdx.x <= 1) {
    args->comm->normTmpStorage[threadIdx.x] = 0;
  }

  __grid_group.sync();

  if(threadIdx.x <= 1) {
      ::atomicAdd(&args->comm->normTmpStorage[threadIdx.x], (float)normShMemory[threadIdx.x]);
  }
  
  __grid_group.sync();

  if (blockIdx.x == 0) {
    prims.sendF32(&args->comm->normTmpStorage[0], 2);

    __syncthreads();

    for (int j=2; j<nranks; ++j) {
        prims.recvReduceSendF32(&args->comm->normTmpStorage[0], 2);
    }

    __syncthreads();

    prims.recvReduceCopyF32(&args->comm->normTmpStorage[0], &args->comm->normTmpStorage[0], 2);
  }

  __grid_group.sync();

  if (threadIdx.x <= 1) {
    normShMemory[threadIdx.x] = sqrtf((float)args->comm->normTmpStorage[threadIdx.x]);
  }

  __syncthreads();

  float wNorm = normShMemory[0];
  float rNorm = normShMemory[1];
  // if (threadIdx.x == 0) {
  //   printf("191: rank %d wwVal %f rrVal %f\n", comm->rank, normShMemory[0], normShMemory[1]);
  // }
  // if(threadIdx.x + blockDim.x*blockIdx.x == 0) args->comm->normTmpStorage = 0;

  // // if (threadIdx.x == 0) {
  // //   printf("rNorm %f\n", (float)rNorm);
  // // }
  // __grid_group.sync();

  // prims.send(&rNorm, 1);

  // for (int j=2; j<nranks; ++j) {
  //     prims.recvReduceSend(&rNorm, 1);
  // }

  // prims.recvReduceCopy(&rNorm, &rNorm, 1);

  // __grid_group.sync();

  // if(threadIdx.x == 0) {
  //     ::atomicAdd(&args->comm->normTmpStorage, (float)rNorm);
  // }
  
  // __grid_group.sync();
  // if(threadIdx.x == 0) {
  //   rNorm = sqrtf((float)args->comm->normTmpStorage);
  // }

  // __syncthreads();
  // // if (threadIdx.x == 0) {
  // //   printf("rNorm %f wNorm %f\n", (float)rNorm, (float)wNorm);
  // // }

  partNum = 0;
  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += nranks*loopSize) {
    int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nranks*args->nChannels));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + bid*nranks*realChunkSize;

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;
    
    chunk = ring->devUserRanks[0];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    prims.computeSend(g + offset, halfw + offset, w, lr, beta1, beta2, m, v, r, offset, nelem, partNum, size, args->nChannels, rNorm, wNorm);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = chunkOffset + chunk * realChunkSize;
      nelem = min(realChunkSize, size-offset);
      prims.directRecvCopySend(halfw + offset, offset, nelem  * 1);
      //prims.directRecvCopySend(thisOutput+offset, offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = chunkOffset + chunk * realChunkSize;
    nelem = min(realChunkSize, size-offset);

    // Final wait/copy.
    prims.directRecv(halfw + offset, offset, nelem);
    //prims.directRecv(thisOutput+offset, offset, nelem);
    partNum++;
  }
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceComputationTreeKernel(struct CollectiveArgs* args) {
  // const int tid = threadIdx.x;
  // const int nthreads = args->nThreads-WARP_SIZE;
  // const int bid = args->bid;
  // struct ncclDevComm* comm = args->comm;
  // struct ncclChannel* channel = comm->channels+blockIdx.x;
  // const ssize_t size = args->N;
  // const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  // int chunkSize = args->lastChunkSize;
  // const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  // const ssize_t loopSize = args->nChannels*chunkSize;

  // if (loopSize > size) {
  //   chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  // }

  // // Compute pointers
  // // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // // T * __restrict__ thisOutput = (T*)args->ThisOutput;
  //   T lr;
  // lr = (T)args->lr;
  // T beta1;
  // beta1 = (T)args->beta1;
  // T beta2;
  // beta2 = (T)args->beta2;
  // const T* __restrict__ g;
  // g = (const T*)args->g;
  // T* __restrict__ w;
  // w = (T*)args->w;
  // T* __restrict__ m;
  // m = (T*)args->m;
  // T* __restrict__ v;
  // v = (T*)args->v;


  // do {
  //   struct ncclTree* tree = &channel->treeUp;
  //   // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
  //   ncclPrimitivesComputation<UNROLL/2, 1, 1, T, NCCL_MAX_TREE_ARITY, 1, FUNC> prims(tid, args->nThreads, tree->down, &tree->up, NULL, stepSize, channel, comm, args->opCount,size,comm->nRanks,args->nChannels, 0);
  //   for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  //     // Up
  //     ssize_t offset = gridOffset + bid*chunkSize;
  //     int nelem = min(chunkSize, size-offset);
  //     if (tree->up == -1) {
  //       prims.recvReduceCopy2(g + offset, w + offset, lr, beta1, beta2, m, v, offset, nelem);
  //       // prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
  //     } else if (tree->down[0] == -1) {
  //       prims.send(g+ offset, nelem);
  //       // prims.send(thisInput+offset, nelem);
  //     } else {
  //       prims.recvReduceSend(g+ offset, nelem);
  //       // prims.recvReduceSend(thisInput+offset, nelem);
  //     }
  //   }
  // } while(0);

  // do {
  //   struct ncclTree* tree = &channel->treeDn;
  //   // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
  //   ncclPrimitivesComputation<UNROLL/2, 1, 1, T, 1, NCCL_MAX_TREE_ARITY, FUNC> prims(tid, args->nThreads, &tree->up, tree->down, NULL, stepSize/*{TREESimple: {stepSize MIXED PRECISION FACTOR}*/, channel, comm, args->opCount,size,comm->nRanks,args->nChannels, 0);
  //   for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  //     // Down
  //     ssize_t offset = gridOffset + bid*chunkSize;
  //     int nelem = min(chunkSize, size-offset);
  //     if (tree->up == -1) {
  //       prims.send(w + offset, nelem);

  //       // prims.send(thisOutput+offset, nelem);
  //     } else if (tree->down[0] == -1) {
  //       prims.recv(w + offset, nelem);

  //       // prims.recv(thisOutput+offset, nelem);
  //     } else {
  //       prims.recvCopySend(w + offset, nelem);

  //       // prims.recvCopySend(thisOutput+offset, nelem);
  //     }
  //   }
  // } while(0);
}

template<int UNROLL, class FUNC, typename T>
__device__ void ncclAllReduceComputationCollNetKernel(struct CollectiveArgs* args) {
  // const int tid = threadIdx.x;
  // const int nthreads = args->nThreads-WARP_SIZE;
  // const int bid = args->bid;
  // struct ncclDevComm* comm = args->comm;
  // struct ncclChannel* channel = comm->channels+blockIdx.x;
  // const ssize_t size = args->N;
  // const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
  // int chunkSize = args->lastChunkSize;
  // const ssize_t minChunkSize = nthreads*8*sizeof(uint64_t) / sizeof(T);
  // const ssize_t loopSize = args->nChannels*chunkSize;

  // if (loopSize > size) {
  //   chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  // }

  // // Compute pointers
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;
  assert(false);
  // if (blockIdx.x < args->nChannels) { // first half of the channels do reduce
  //   struct ncclTree* tree = &channel->collTreeUp;
  //   ncclPrimitivesComputation<UNROLL, 1, 1, T, 1, 1, FUNC> prims(tid, args->nThreads, tree->down, &tree->up, NULL, stepSize, channel, comm, args->opCount);
  //   for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  //     // Up
  //     ssize_t offset = gridOffset + bid*chunkSize;
  //     int nelem = min(chunkSize, size-offset);
  //     if (tree->up == -1) {
  //       prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
  //     } else if (tree->down[0] == -1) {
  //       prims.send(thisInput+offset, nelem);
  //     } else {
  //       prims.recvReduceSend(thisInput+offset, nelem);
  //     }
  //   }
  // }

  // if (blockIdx.x >= args->nChannels) { // second half of the channels do broadcast
  //   struct ncclTree* tree = &channel->collTreeDn;
  //   ncclPrimitivesComputation<UNROLL, 1, 1, T, 1, 1, FUNC> prims(tid, args->nThreads, &tree->up, tree->down, NULL, stepSize, channel, comm, args->opCount);
  //   for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  //     // Down
  //     ssize_t offset = gridOffset + bid*chunkSize;
  //     int nelem = min(chunkSize, size-offset);
  //     if (tree->up == -1) {
  //       prims.send(thisOutput+offset, nelem);
  //     } else if (tree->down[0] == -1) {
  //       prims.recv(thisOutput+offset, nelem);
  //     } else {
  //       prims.recvCopySend(thisOutput+offset, nelem);
  //     }
  //   }
  // }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationRingLLKernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->N;
  const int nranks = comm->nRanks;

  //const int rank = comm->rank;
  ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);

  const ssize_t loopSize = args->nChannels*nranks*chunkSize;
  const ssize_t maxPartSize = min(DIVUP(size-0, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

  ncclLLPrimitivesComputation<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount, size, nranks, args->nChannels, loopSize, maxPartSize);

  // Compute pointers
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;
  float lr = (float)args->lr;
  float beta1;
  beta1 = (float)args->beta1;
  float beta2;
  beta2 = (float)args->beta2;
  const T* __restrict__ g;
  g = (const T*)args->g;
  T* __restrict__ halfw;
  halfw = (T*)args->halfw;
  float* __restrict__ r = (float*)args->r;;
  float* __restrict__ w;
  w = (float*)args->w;
  float* __restrict__ m;
  m = (float*)args->m;
  float* __restrict__ v;
  v = (float*)args->v;
  float lambda = (float)1.0f;

  __shared__ float normShMemory[2];

  if (threadIdx.x <= 1) {
    normShMemory[threadIdx.x] = 0.0f;
  }

  __syncthreads();

  cooperative_groups::grid_group __grid_group = cooperative_groups::this_grid();

  register float wwVal = 0;
  register float rrVal = 0;

  int partNum = 0;
  /*RINGLL: {INSERT SHARED MEMORY FOR REDUCTION}*/
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
   
    LLprims.send(g+ offset, nelem);
    //LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);

      LLprims.recvReduceSend(g+ offset, nelem);
      //LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);
    // if (threadIdx.x == 0) {
    //   printf("332: minChunkSize %ld chunkSize %ld nelem %d partNum %d blockIdx.x %d args->nChannels %d loopSize %ld offset %ld rank %d\n", 
    //          minChunkSize, chunkSize, nelem, partNum, blockIdx.x, args->nChannels, loopSize, offset, comm->rank);
    // }

    LLprims.recvReduceCopy2(lr, beta1, beta2, g + offset, halfw + offset, w + offset, m + offset, v + offset, r + offset, offset, nelem, (float*)&rrVal, (float*)&wwVal);
  }  

  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    wwVal += __shfl_down_sync(0xffffffff, wwVal, offset);
    rrVal += __shfl_down_sync(0xffffffff, rrVal, offset);
  }

  if (threadIdx.x % warpSize == 0)
    ::atomicAdd((float*)(&normShMemory[0]), (float)(wwVal)); 

  if (threadIdx.x % warpSize == 0)
    ::atomicAdd((float*)(&normShMemory[1]), (float)(rrVal)); 

  __syncthreads();
  // if(threadIdx.x == 0) {
  //   printf("522: rank: %d blockIdx.x %d glmwNorm %f glmrNorm %f shwNorm %f shrNorm %f wwVal %f rrVal %f\n",  comm->rank, blockIdx.x, 
  //   args->comm->normTmpStorage[0], args->comm->normTmpStorage[1], normShMemory[0], normShMemory[1], wwVal, rrVal);
  // }

  if(threadIdx.x + blockDim.x*blockIdx.x <= 1) {
    args->comm->normTmpStorage[threadIdx.x] = 0;
  }

  __grid_group.sync();

  if(threadIdx.x <= 1) {
      ::atomicAdd(&args->comm->normTmpStorage[threadIdx.x], (float)normShMemory[threadIdx.x]);
  }

  __grid_group.sync();

  // if(threadIdx.x == 0) {
  //   printf("522: rank: %d blockIdx.x %d glmwNorm %f glmrNorm %f shwNorm %f shrNorm %f wwVal %f rrVal %f\n",  comm->rank, blockIdx.x, 
  //   args->comm->normTmpStorage[0], args->comm->normTmpStorage[1], normShMemory[0], normShMemory[1], wwVal, rrVal);
  // }


  float* normStorage = (float*)&args->comm->normTmpStorage;
  const ssize_t normSize = 2;
  chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(float);
  const ssize_t normLoopSize = args->nChannels*nranks*chunkSize;
  for (ssize_t gridOffset = 0; gridOffset < normSize; gridOffset += normLoopSize) {
    chunkSize = min(DIVUP(normSize-gridOffset, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, normSize-offset);

    LLprims.sendF32(normStorage+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, normSize-offset);

      LLprims.recvReduceSendF32(normStorage+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, normSize-offset);

    LLprims.recvReduceCopySendF32(normStorage+offset, normStorage+offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, normSize-offset);

      LLprims.recvCopySendF32(normStorage+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, normSize-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recvF32(normStorage+offset, nelem);
  }
  __grid_group.sync();

  if (threadIdx.x <= 1) {
    normShMemory[threadIdx.x] = sqrtf((float)args->comm->normTmpStorage[threadIdx.x]);
  }

  __syncthreads();

  // if(threadIdx.x == 0) {
  //   printf("rank: %d glmwNorm %f glmrNorm %f\n",  comm->rank, 
  //   normShMemory[0], normShMemory[1]);
  // }

  float wNorm = (float)normShMemory[0];
  float rNorm = (float)normShMemory[1];

  float scale = ((wNorm > 0) ? (rNorm > 0 ? wNorm/rNorm : 1.0f) : 1.0f)/rNorm;
  
  // if (threadIdx.x == 0) {
  //   printf("scale %f\n", scale);
  // }
 
  chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
 
  partNum = 0;
  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);
    
    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.sendLAMB(lr, g + offset, halfw+offset, w + offset, r+offset, nelem, scale);
    
    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);
     LLprims.recvCopySend(halfw + offset,  nelem);
      //LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(halfw + offset,  nelem);
    //LLprims.recv(thisOutput+offset, nelem);

    partNum++;
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationTreeLLKernel(struct CollectiveArgs* args) {
#if 0
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
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;
    T lr;
  lr = (T)args->lr;
  T beta1;
  beta1 = (T)args->beta1;
  T beta2;
  beta2 = (T)args->beta2;
  const T* __restrict__ g;
  g = (const T*)args->g;
  T* __restrict__ w;
  w = (T*)args->w;
  T* __restrict__ m;
  m = (T*)args->m;
  T* __restrict__ v;
  v = (T*)args->v;

  /*TREELL: {REDUCTION SHMEM}*/
  
  do {
    struct ncclTree* tree = &channel->treeUp;
    // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
    ncclLLPrimitivesComputation<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreads, tree->down, &tree->up, channel, comm, args->opCount,size,comm->nRanks,0);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Up
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.recvReduceCopy2(lr, beta1, beta2, g + offset, w + offset, m, v, offset, nelem);
        //LLprims.recvReduceCopy2(thisInput+offset, thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.send(g+ offset, nelem);
        //LLprims.send(thisInput+offset, nelem);
      } else {
        LLprims.recvReduceSend(g+ offset, nelem);
        //LLprims.recvReduceSend(thisInput+offset, nelem);
      }
    }

    if (tree->up == -1) {
      /*TREELL: REDUCTION {REDUCTION COMPUTATION}*/

      /*TREELL: REDUCTION {GLOBAL MEM REDUCTION}*/
      /*TREELL: REDUCTION {COMPUTATION}*/
    }
  } while(0);

  do {
    struct ncclTree* tree = &channel->treeDn;
    // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
    ncclLLPrimitivesComputation<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, &tree->up, tree->down, channel, comm, args->opCount, 0, 0,0);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      // Down
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      if (tree->up == -1) {
        LLprims.send(w + offset, nelem);

        //LLprims.send(thisOutput+offset, nelem);
      } else if (tree->down[0] == -1) {
        LLprims.recv(w + offset, nelem);

        //LLprims.recv(thisOutput+offset, nelem);
      } else {
        LLprims.recvCopySend(w + offset, nelem);

        //LLprims.recvCopySend(thisOutput+offset, nelem);
      }
    }
  } while(0);

#endif
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationCollNetLLKernel(struct CollectiveArgs* args) {
  assert(false);
  // const int tid = threadIdx.x;
  // const int nthreads = args->nThreads;
  // const int bid = args->bid;
  // struct ncclDevComm* comm = args->comm;
  // struct ncclChannel* channel = comm->channels+blockIdx.x;
  // const ssize_t size = args->N;
  // ssize_t chunkSize = NCCL_LL_SLICE_LINES * sizeof(uint64_t) / sizeof(T);
  // const ssize_t minChunkSize = nthreads*sizeof(uint64_t) / sizeof(T);
  // const ssize_t loopSize = args->nChannels*chunkSize;

  // if (loopSize > size) {
  //   chunkSize = DIVUP(size, args->nChannels*minChunkSize)*minChunkSize;
  // }

  // // Compute pointers
  // const T * __restrict__ thisInput = (const T*)args->ThisInput;
  // T * __restrict__ thisOutput = (T*)args->ThisOutput;

  // if (blockIdx.x < args->nChannels) { // first half of the channels do reduce
  //   struct ncclTree* tree = &channel->collTreeUp;
  //   ncclLLPrimitivesComputation<T, FUNC, 1, 1> LLprims(tid, nthreads, tree->down, &tree->up, channel, comm, args->opCount);
  //   for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  //     // Up
  //     ssize_t offset = gridOffset + bid*chunkSize;
  //     int nelem = min(chunkSize, size-offset);
  //     if (tree->up == -1) {
  //       LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
  //     } else if (tree->down[0] == -1) {
  //       LLprims.send(thisInput+offset, nelem);
  //     } else {
  //       LLprims.recvReduceSend(thisInput+offset, nelem);
  //     }
  //   }
  // }

  // if (blockIdx.x >= args->nChannels) { // second half of the channels do broadcast
  //   struct ncclTree* tree = &channel->collTreeDn;
  //   ncclLLPrimitivesComputation<T, FUNC, 1, 1> LLprims(tid, nthreads, &tree->up, tree->down, channel, comm, args->opCount);
  //   for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
  //     // Down
  //     ssize_t offset = gridOffset + bid*chunkSize;
  //     int nelem = min(chunkSize, size-offset);
  //     if (tree->up == -1) {
  //       LLprims.send(thisOutput+offset, nelem);
  //     } else if (tree->down[0] == -1) {
  //       LLprims.recv(thisOutput+offset, nelem);
  //     } else {
  //       LLprims.recvCopySend(thisOutput+offset, nelem);
  //     }
  //   }
  // }
}

#include "prims_ll128_computation.h"
template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationRingLL128Kernel(struct CollectiveArgs* args) {
  const int tid = threadIdx.x;
  const int bid = args->bid;
  const int nthreads = args->nThreads;
  struct ncclDevComm* comm = args->comm;
  struct ncclChannel* channel = comm->channels+blockIdx.x;
  struct ncclRing* ring = &channel->ring;
  const ssize_t size = args->N;
  //const int rank = comm->rank;
  const int nranks = comm->nRanks;

  ssize_t chunkSize = (NCCL_LL128_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));

  // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
  const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;

  const ssize_t loopSize = args->nChannels*nranks*chunkSize;
  const ssize_t maxPartSize = min(DIVUP(size-0, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

  ncclLL128PrimitivesComputation<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, channel, comm, args->opCount, args->N, nranks, args->nChannels, loopSize, maxPartSize);

  

  // Compute pointers
  float lr;
  lr = (float)args->lr;
  float beta1;
  beta1 = (float)args->beta1;
  float beta2;
  beta2 = (float)args->beta2;
  const T* __restrict__ g;
  g = (const T*)args->g;
  T* __restrict__ halfw;
  halfw = (T*)args->halfw;
  float* __restrict__ r = (float*)args->r;;
  float* __restrict__ w;
  w = (float*)args->w;
  float* __restrict__ m;
  m = (float*)args->m;
  float* __restrict__ v;
  v = (float*)args->v;
  float lambda = (float)1.0f;

  __shared__ float normShMemory[2];


  if (threadIdx.x <= 1) {
    normShMemory[threadIdx.x] = 0.0f;
  }

  __syncthreads();

  cooperative_groups::grid_group __grid_group = cooperative_groups::this_grid();

  register float wwVal = 0;
  register float rrVal = 0;

  int partNum = 0;

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
    LLprims.send(g+ offset, nelem);
    //LLprims.send(thisInput+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);
      LLprims.recvReduceSend(g+ offset, nelem);
      //LLprims.recvReduceSend(thisInput+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // if (threadIdx.x == 0) {
    //   printf("332: minChunkSize %ld chunkSize %ld nelem %d partNum %d blockIdx.x %d args->nChannels %d loopSize %ld offset %ld rank %d nthreads %d\n", 
    //          minChunkSize, chunkSize, nelem, partNum, blockIdx.x, args->nChannels, loopSize, offset, comm->rank, nthreads);
    // }

    LLprims.recvReduceCopy2(lr, beta1, beta2, g + offset, halfw + offset, w + offset, m+ offset, v+ offset, r+offset,offset, partNum, 
                            nelem, &wwVal, &rrVal);
    //LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
  }

  for (int offset = warpSize/2; offset > 0; offset /= 2) {
    wwVal += __shfl_down_sync(0xffffffff, wwVal, offset);
    rrVal += __shfl_down_sync(0xffffffff, rrVal, offset);
  }

  if (threadIdx.x % warpSize == 0)
    ::atomicAdd((float*)(&normShMemory[0]), (float)(wwVal)); 

  if (threadIdx.x % warpSize == 0)
    ::atomicAdd((float*)(&normShMemory[1]), (float)(rrVal)); 

  __syncthreads();


  if(threadIdx.x + blockDim.x*blockIdx.x <= 1) {
    args->comm->normTmpStorage[threadIdx.x] = 0;
  }

  __grid_group.sync();

  if(threadIdx.x <= 1) {
      ::atomicAdd(&args->comm->normTmpStorage[threadIdx.x], (float)normShMemory[threadIdx.x]);
  }

  __grid_group.sync();

  // if(threadIdx.x == 0) {
  //   printf("522: rank: %d blockIdx.x %d glmwNorm %f glmrNorm %f shwNorm %f shrNorm %f wwVal %f rrVal %f\n",  comm->rank, blockIdx.x, 
  //   args->comm->normTmpStorage[0], args->comm->normTmpStorage[1], normShMemory[0], normShMemory[1], wwVal, rrVal);
  // }

  float* normStorage = (float*)&args->comm->normTmpStorage;
  const ssize_t normSize = 2;
  chunkSize = (NCCL_LL128_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));
  const ssize_t normLoopSize = args->nChannels*nranks*chunkSize;
  for (ssize_t gridOffset = 0; gridOffset < normSize; gridOffset += normLoopSize) {
    chunkSize = min(DIVUP(normSize-gridOffset, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = ring->devUserRanks[nranks-1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, normSize-offset);

    LLprims.sendF32(normStorage+offset, nelem);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, normSize-offset);

      LLprims.recvReduceSendF32(normStorage+offset, nelem);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, normSize-offset);

    LLprims.recvReduceCopySendF32(normStorage+offset, normStorage+offset, nelem);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, normSize-offset);

      LLprims.recvCopySendF32(normStorage+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, normSize-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recvF32(normStorage+offset, nelem);
  }

  __grid_group.sync();

  if (threadIdx.x <= 1) {
    normShMemory[threadIdx.x] = sqrtf((float)args->comm->normTmpStorage[threadIdx.x]);
  }

  __syncthreads();


  // if(threadIdx.x == 0) {
  //   printf("909: rank: %d blockIdx.x %d glmwNorm %f glmrNorm %f shwNorm %f shrNorm %f wwVal %f rrVal %f\n",  comm->rank, blockIdx.x, 
  //   args->comm->normTmpStorage[0], args->comm->normTmpStorage[1], normShMemory[0], normShMemory[1], wwVal, rrVal);
  // }

  float wNorm = (float)normShMemory[0];
  float rNorm = (float)normShMemory[1];

  float scale = ((wNorm > 0) ? (rNorm > 0 ? wNorm/rNorm : 1.0f) : 1.0f)/rNorm;
  
  chunkSize = (NCCL_LL128_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    chunkSize = min(DIVUP(size-gridOffset, args->nChannels*nranks*minChunkSize)*minChunkSize, chunkSize);

    /////////////// begin AllReduce steps ///////////////
    ssize_t offset;
    int nelem;
    int chunk;

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ring->devUserRanks[0];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    LLprims.sendLAMB(lr, g + offset, halfw + offset, w + offset, r + offset, nelem, offset, (T)scale);

    // k-2 steps: copy to next GPU
    for (int j=1; j<nranks-1; ++j) {
      chunk = ring->devUserRanks[nranks-j];
      offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
      nelem = min(chunkSize, size-offset);
      LLprims.recvCopySend(halfw + offset,  nelem);
      //LLprims.recvCopySend(thisOutput+offset, nelem);
    }

    // Make final copy from buffer to dest.
    chunk = ring->devUserRanks[1];
    offset = gridOffset + (chunk*args->nChannels+bid) * chunkSize;
    nelem = min(chunkSize, size-offset);

    // Here we need to copy from buffer to this output.
    LLprims.recv(halfw + offset,  nelem);
    //LLprims.recv(thisOutput+offset, nelem);

    partNum += 1;
  }
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationTreeLL128Kernel(struct CollectiveArgs* args) {
  #if 0
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
    T lr;
  lr = (T)args->lr;
  T beta1;
  beta1 = (T)args->beta1;
  T beta2;
  beta2 = (T)args->beta2;
  const T* __restrict__ g;
  g = (const T*)args->g;
  T* __restrict__ w;
  w = (T*)args->w;
  T* __restrict__ m;
  m = (T*)args->m;
  T* __restrict__ v;
  v = (T*)args->v;

  if (treeUp->up == -1) {
    // ReduceAndBroadcast : max number of recv is 3, max number of send is 3
    ncclLL128PrimitivesComputation<T, FUNC, NCCL_MAX_TREE_ARITY, NCCL_MAX_TREE_ARITY> LLprims(tid, nthreads, treeUp->down, treeDn->down, channel, comm, args->opCount, size, 0, 0);
    for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
      ssize_t offset = gridOffset + bid*chunkSize;
      int nelem = min(chunkSize, size-offset);
      LLprims.recvReduceCopySend(lr, beta1, beta2, g + offset, w + offset, m + offset, v + offset, offset, 0, nelem);
      //LLprims.recvReduceCopySend(thisInput+offset, thisOutput+offset, nelem);
    }
  } else {
    if (tid < nthreadsSplit) {
      // Reduce : max number of recv is 3, max number of send is 1 (binary tree + local)
      ncclLL128PrimitivesComputation<T, FUNC, NCCL_MAX_TREE_ARITY, 1> LLprims(tid, nthreadsSplit, treeUp->down, &treeUp->up, channel, comm, args->opCount, size, 0, 0);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Up
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeUp->down[0] == -1) {
          LLprims.send(g+ offset, nelem);
          // LLprims.send(thisInput+offset, nelem);
        } else {
          LLprims.recvReduceSend(g+ offset, nelem);
          // LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    } else {
      // Broadcast : max number of recv is 1, max number of send is 3 (binary tree + local)
      ncclLL128PrimitivesComputation<T, FUNC, 1, NCCL_MAX_TREE_ARITY> LLprims(tid-nthreadsSplit, nthreads-nthreadsSplit, &treeDn->up, treeDn->down, channel, comm, args->opCount, size, 0, 0);
      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        // Down
        ssize_t offset = gridOffset + bid*chunkSize;
        int nelem = min(chunkSize, size-offset);
        if (treeDn->down[0] == -1) {
          LLprims.recv(w + offset,  nelem);
          // LLprims.recv(thisOutput+offset, nelem);
        } else {
          LLprims.recvCopySend(w + offset,  nelem);
          // LLprims.recvCopySend(thisOutput+offset, nelem);
        }
      }
    }
  }
  #endif
}

template<int UNUSED, class FUNC, typename T>
__device__ void ncclAllReduceComputationCollNetLL128Kernel(struct CollectiveArgs* args) {assert(false); }

#endif 