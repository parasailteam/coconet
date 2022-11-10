/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include <assert.h>

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, recvbuff, NULL, NULL, NULL, nullptr, nullptr, nullptr, nullptr, 0, 
    /*ReduceScattered Params*/ nullptr, nullptr, nullptr, nullptr, 0, nullptr, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceScatteredWeights, const void* sendbuff, void** weightbuff,
    size_t nbuff, size_t* counts, size_t totalCount,  void* alpha, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceScatteredWeights(const void* sendbuff, void** weightbuff, size_t nbuff, size_t* counts, size_t totalCount,  void* alpha,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, nullptr, nullptr, nullptr, alpha, nullptr, nullptr, nullptr, nullptr, 0, 
    /*ReduceScattered Params*/ nullptr, weightbuff, nullptr, nullptr, nbuff, counts,
    /*Other args*/ totalCount, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceScatteredGradWeights, const void** sendbuff, void** weightbuff,
    size_t nbuff, size_t* counts, size_t totalCount,  void* alpha, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceScatteredGradWeights(const void** sendbuff, void** weightbuff, size_t nbuff, size_t* counts, size_t totalCount,  void* alpha,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {

  struct ncclInfo info = { ncclCollAllReduceScattered, "AllReduceScattered",
    nullptr, nullptr, nullptr, nullptr, alpha, nullptr, nullptr, nullptr, nullptr, 0, 
    /*ReduceScattered Params*/ sendbuff, weightbuff, nullptr, nullptr, nbuff, counts,
    /*Other args*/ totalCount, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceScatteredAdam, const void** sendbuff, void** weightbuff, void* firstMomentBuff, void* secondMomentBuff,
    size_t nbuff, size_t* counts, size_t totalCount, void* alpha, void* beta1Buff, void* beta2Buff, const int epoch, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceScatteredAdam(const void** sendbuff, void** weightbuff, void* firstMomentBuff, void* secondMomentBuff, size_t nbuff, size_t* counts, size_t totalCount,
    void* alpha, void* beta1Buff, void* beta2Buff, const int epoch,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollAllReduceScattered, "AllReduceScattered",
    nullptr, nullptr, nullptr, nullptr, alpha, nullptr, nullptr, 
    beta1Buff, beta2Buff, epoch, 
    /*ReduceScattered Params*/ sendbuff, weightbuff, firstMomentBuff, secondMomentBuff, nbuff, counts,
    totalCount, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, OptimizerType::Adam, nullptr };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceScatteredLAMB, const void** sendbuff, void** weightbuff, void* firstMomentBuff, void* secondMomentBuff,
    size_t smallNbuff, size_t* counts, size_t totalCount,  const size_t* parentBuffSizes, void* alpha, void* beta1Buff, void* beta2Buff, void* weightNormBuff, void* rStorageBuff, 
    const size_t nbuff, size_t* buffIdToParentBufferId, const int epoch, ncclDataType_t datatype, 
    ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceScatteredLAMB(const void** sendbuff, void** weightbuff, void* firstMomentBuff, void* secondMomentBuff, size_t smallNbuff, size_t* counts, size_t totalCount,  const size_t* parentBuffSizes,
    void* alpha, void* beta1Buff, void* beta2Buff, void* weightNormBuff, void* rStorageBuff, const size_t nBuff, size_t* buffIdToParentBufferId, const int epoch,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollAllReduceScattered, "AllReduceScattered",
    nullptr, nullptr, nullptr, nullptr, alpha, nullptr, nullptr, 
    beta1Buff, beta2Buff, epoch, 
    /*ReduceScattered Params*/ sendbuff, weightbuff, firstMomentBuff, secondMomentBuff, smallNbuff, counts,
    totalCount, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, OptimizerType::LAMB, weightNormBuff, nBuff, buffIdToParentBufferId, rStorageBuff, parentBuffSizes};
  return ncclEnqueueCheck(&info);
}


NCCL_API(ncclResult_t, ncclAllReduceDropoutBiasLayernorm, const void* sendbuff, void* recvbuff,
    void* bias, void* addTensor, size_t count, size_t biasLength, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceDropoutBiasLayernorm(const void* sendbuff, void* recvbuff,void* bias, void* addTensor, size_t count, size_t biasLength,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream)    {
      struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, recvbuff, nullptr, nullptr, nullptr, bias, addTensor,  nullptr, nullptr, (int)biasLength, 
    /*ReduceScattered Params*/ nullptr, nullptr, nullptr, nullptr, 0, nullptr,
    count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, OptimizerType::DropoutBiasLayerNorm};

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceOverlappedMatMulDropoutBiasLayernorm, const void* m1buff, void* m2buff, void* m1m2buff,
    void* syncGlobalMem, void* bias, void* addTensor, size_t count, int M, int N, int K, size_t biasLength, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceOverlappedMatMulDropoutBiasLayernorm(const void* m1buff, void* m2buff, void* m1m2buff, void* syncGlobalMem, 
    void* bias, void* addTensor, size_t count, int M, int N, int K, size_t biasLength,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream)    {
  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
  m1buff, m2buff, m1m2buff, syncGlobalMem, nullptr, bias, addTensor,  nullptr, nullptr, (int)biasLength, 
  /*ReduceScattered Params*/ nullptr, nullptr, nullptr, nullptr, 0, nullptr,
  count, datatype, op, 0, comm, stream, /* Args */
  ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, OptimizerType::DropoutBiasLayerNorm};
  struct MatMulConfig matMulConfig = {M, N, K};
  info.matMulConfig = matMulConfig;

  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceMatrix, void* m1m2buff, size_t count, int M, int N, int realChunkCols, ncclDataType_t datatype, 
ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceMatrix(void* m1m2buff, size_t count, int M, int N,
    int realChunkCols,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct MatMulConfig matMulConfig = {M, N, -1, realChunkCols};

  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    m1m2buff, m1m2buff, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 
    /*ReduceScattered Params*/ nullptr, nullptr, nullptr, nullptr, 0, nullptr,
    count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  info.matMulConfig = matMulConfig;
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceOverlapMatMul, const void* m1buff, void* m2buff, void* m1m2buff,
    void* syncGlobalMem, size_t count, int M, int N, int K, int realChunkCols, 
    int outerIteration, 
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceOverlapMatMul(const void* m1buff, void* m2buff, void* m1m2buff,
    void* syncGlobalMem, size_t count, int M, int N, int K, int realChunkCols, 
    int outerIteration,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct MatMulConfig matMulConfig = {M, N, K, realChunkCols, outerIteration, 24};

  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    m1buff, m2buff, m1m2buff, syncGlobalMem, nullptr, nullptr, nullptr, nullptr, nullptr, 0, 
    /*ReduceScattered Params*/ nullptr, nullptr, nullptr, nullptr, 0, nullptr,
    count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  info.matMulConfig = matMulConfig;
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclAllReduceAdam, const void* sendbuff, void* recvbuff, void* weightbuff,
    size_t count,  void* alpha, void* firstMomentBuff, void* secondMomentBuff, void* beta1Buff, void* beta2Buff,
    const int epoch, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduceAdam(const void* sendbuff, void* recvbuff, void* weightbuff, size_t count, void* alpha,
    void* firstMomentBuff, void* secondMomentBuff, void* beta1Buff, void* beta2Buff, const int epoch, ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  
  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, nullptr, weightbuff, nullptr, alpha, firstMomentBuff, secondMomentBuff, 
    beta1Buff, beta2Buff, epoch, 
    /*ReduceScattered Params*/ nullptr, nullptr, nullptr, nullptr, 0, nullptr,
    count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}