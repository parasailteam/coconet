/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
    struct ncclFusedComputationParams params;
  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, params };
  return ncclEnqueueCheck(&info);
}

#define TYPE_ALL_REDUCE 1

#if TYPE_ALL_REDUCE == 0
NCCL_API(ncclResult_t, AllReduce_pipe, float lr, float beta1, float beta2, half* g, float* w, half* halfw, float* m, float* v, size_t count, ncclDataType_t datatype, ncclComm_t comm, ncclRedOp_t op, cudaStream_t stream);

ncclResult_t AllReduce_pipe(float lr, float beta1, float beta2, half* g, float* w, half* halfw,float* m, float* v, size_t count, ncclDataType_t datatype, ncclComm_t comm, ncclRedOp_t op, cudaStream_t stream){
struct ncclFusedComputationParams fusedComputationParams = {lr, beta1, beta2, g, w, halfw, m, v, nullptr};
struct ncclInfo info = {ncclCollAllReduceComputation, "AllReduceComputation", nullptr, nullptr, count, datatype, op, 0,comm, stream, ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, fusedComputationParams };
  return ncclEnqueueCheck(&info);
}

#elif TYPE_ALL_REDUCE == 1

NCCL_API(ncclResult_t, AllReduce_pipe, float lr, float beta1, float beta2, half* g, float* w, half* halfw, float* m, float* v, float* r, size_t count, ncclDataType_t datatype, ncclComm_t comm, ncclRedOp_t op, cudaStream_t stream);

ncclResult_t AllReduce_pipe(float lr, float beta1, float beta2, half* g, float* w, half* halfw,float* m, float* v, float* r, size_t count, ncclDataType_t datatype, ncclComm_t comm, ncclRedOp_t op, cudaStream_t stream){
struct ncclFusedComputationParams fusedComputationParams = {lr, beta1, beta2, g, w, halfw, m, v, r};
struct ncclInfo info = {ncclCollAllReduceComputation, "AllReduceComputation", nullptr, nullptr, count, datatype, op, 0,comm, stream, ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS, fusedComputationParams };
  return ncclEnqueueCheck(&info);
}
#endif