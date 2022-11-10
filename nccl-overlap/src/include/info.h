/*************************************************************************
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "core.h"

typedef enum {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
  ncclPatternCollTreeUp,
  ncclPatternCollTreeDown
} ncclPattern_t;

struct MatMulConfig {
  int MATMUL_M;
  int MATMUL_N;
  int MATMUL_K;
  int realChunkCols;
  int outerIteration;
  int combinedChunks;
};

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  void* weightbuff;
  void* syncGlobalMem;
  void* alpha;
  //Adam Parameters
  void* firstMomentBuff; 
  void* secondMomentBuff;
  void* beta1Buff;
  void* beta2Buff;
  int epoch;
  //Scattered Pointer Params
  const void** scatteredSendbuff;
  void** scatteredWeightbuff;
  void* scatteredFirstMomentBuff;
  void* scatteredSecondMomentBuff;
  size_t scatteredSmallNBuff;
  size_t* scatteredBuffSizes;
  ///
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root;
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;

  OptimizerType optimizerType;
  void* weightNormBuff;
  size_t nbuff;
  size_t* buffIdToParentBufferId;
  void* rStorageBuff;
  const size_t* parentBuffSizes;
  

  // Computed later
  int algorithm;
  int protocol;
  ncclPattern_t pattern;
  int nChannels;
  int nThreads;
  size_t nBytes;
  int nstepsPerLoop;
  int nchunksPerLoop;


  MatMulConfig matMulConfig;

};

#endif
