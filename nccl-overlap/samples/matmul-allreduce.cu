//nvcc matmul-allreduce.cu -std=c++11 -Xcompiler -fopenmp,-O3 -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/mpi -I.. -I/usr/local/cuda/include/ -I ../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c && mpicxx matmul-allreduce.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o matmul-allreduce

#include "header.h"
#include "cutlass-matmul.h"
#include <cuda_profiler_api.h>

#include <map> 

void pipe_columnmajorABC(cublasHandle_t handle, const half *alpha, const half *beta, const half* m1, const half* m2, half* m1m2, ncclComm_t comm, cudaStream_t stream, int M, int N, int K, float& allReduceTime, float& cublasTime) {
  cudaEvent_t startpipe, stoppipe;
    float elapsedTime = 0;
    // MPI_Barrier(MPI_COMM_WORLD);

    CUDACHECK(cudaEventCreate(&startpipe));
    CUDACHECK(cudaEventCreate(&stoppipe));
    CUDACHECK(cudaEventRecord(startpipe, stream));
  CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    M, N, K, 
    alpha,
    m1, CUDA_R_16F, M,
    m2, CUDA_R_16F, K,
    beta, 
    m1m2, CUDA_R_16F, M,
    CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    CUDACHECK(cudaEventRecord(stoppipe, stream));

    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaEventSynchronize(stoppipe));
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

    cublasTime += elapsedTime;

  elapsedTime = 0;
  CUDACHECK(cudaEventCreate(&startpipe));
  CUDACHECK(cudaEventCreate(&stoppipe));
  CUDACHECK(cudaEventRecord(startpipe, stream));

  NCCLCHECK(ncclAllReduceMatrix(m1m2, M*N, M, N, N, ncclHalf, ncclSum, comm, stream));

  CUDACHECK(cudaEventRecord(stoppipe, stream));
  CUDACHECK(cudaEventSynchronize(stoppipe));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

  allReduceTime += elapsedTime;

}


void pipe_rowmajorABC(cublasHandle_t handle, const half *alpha, const half *beta, const half* m1, const half* m2, half* m1m2, ncclComm_t comm, cudaStream_t stream, int M, int N, int K, float& allReduceTime, float& cublasTime) {
  cudaEvent_t startpipe, stoppipe;
    float elapsedTime = 0;
    // MPI_Barrier(MPI_COMM_WORLD);

    CUDACHECK(cudaEventCreate(&startpipe));
    CUDACHECK(cudaEventCreate(&stoppipe));
    CUDACHECK(cudaEventRecord(startpipe, stream));
  CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    N, M, K, 
    alpha,
    m2, CUDA_R_16F, N,
    m1, CUDA_R_16F, K,
    beta, 
    m1m2, CUDA_R_16F, N,
    CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    CUDACHECK(cudaEventRecord(stoppipe, stream));

    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaEventSynchronize(stoppipe));
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

    cublasTime += elapsedTime;

  elapsedTime = 0;
  double t1 = getCurrentTime();

  NCCLCHECK(ncclAllReduceMatrix(m1m2, M*N, M, N, N, ncclHalf, ncclSum, comm, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  double t2 = getCurrentTime();
  allReduceTime += (t2-t1)*1000.0f;

}

const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;
#include<cuda_fp16.h>
#include<mma.h>
using namespace nvcuda;
// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16. 
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
 __global__ void customMatMulKernel(const half *a, const half *b, half *c, int M, int N, int K, float alpha, float beta) {
  // Leading dimensions. Packed with no transpositions.
  int lda = M;
  int ldb = K;
  int ldc = M;

  // Tile using a 2D grid
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> acc_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < K; i += WMMA_K) {
  int aRow = warpM * WMMA_M;
  int aCol = i;

  int bRow = i;
  int bCol = warpN * WMMA_N;

  // Bounds checking
  if (aRow < M && aCol < K && bRow < K && bCol < N) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

  }
  }

  // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
  int cRow = warpM * WMMA_M;
  int cCol = warpN * WMMA_N;

  if (cRow < M && cCol < N) {
  //    wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);


  for(int i=0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = acc_frag.x[i];
  }

  // Store the output
  wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
  }
}

// Fused Version
void pipe_customMatMulKernel(cublasHandle_t handle, const half *alpha, const half *beta, const half* m1, const half* m2, half* m1m2, ncclComm_t comm, cudaStream_t stream, int M, int N, int K, float& allReduceTime, float& matmulTime) { 
  cudaEvent_t startpipe, stoppipe;
  float elapsedTime = 0;
  // MPI_Barrier(MPI_COMM_WORLD);

  CUDACHECK(cudaEventCreate(&startpipe));
  CUDACHECK(cudaEventCreate(&stoppipe));
  CUDACHECK(cudaEventRecord(startpipe, stream));
  dim3 gridDim;
  dim3 blockDim;

  // blockDim.x must be a multple of warpSize
  // 128x4 means we have 16 warps and a block computes a 64x64 output tile
  blockDim.x = 128;
  blockDim.y = 4;

  gridDim.x = (M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
  gridDim.y = (N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
  customMatMulKernel<<<gridDim, blockDim, 0, stream>>> (m1, m2, m1m2, M, N, K, 1, 0);
  CUDACHECK(cudaEventRecord(stoppipe, stream));

  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaEventSynchronize(stoppipe));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

  matmulTime += elapsedTime;

  elapsedTime = 0;
  CUDACHECK(cudaEventCreate(&startpipe));
  CUDACHECK(cudaEventCreate(&stoppipe));
  CUDACHECK(cudaEventRecord(startpipe, stream));

  NCCLCHECK(ncclAllReduce(m1m2, m1m2, M*N, ncclHalf, ncclSum, comm, stream));

  CUDACHECK(cudaEventRecord(stoppipe, stream));
  CUDACHECK(cudaEventSynchronize(stoppipe));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

  allReduceTime += elapsedTime;
}

// Fused Version
void pipe_fuseed(cublasHandle_t handle, const half *alpha, const half *beta, const half* m1, const half* m2, half* m1m2, ncclComm_t comm, cudaStream_t stream, int M, int N, int K, float& allReduceTime, float& cublasTime) {

  NCCLCHECK(ncclAllReduceOverlapMatMul((const void*)m1, (void*)m2, (void*)m1m2, nullptr, M*N, M, N, K, N, 0, ncclHalf, ncclSum, comm, stream));
}


int userRankForChannels[80];

#if 0
// void PREVIOUS_pipe1_cublas_overlapped(int iter, cublasHandle_t* handles, cudaStream_t* streams, int* syncGlobalMem, int nChannels, 
//   std::vector<std::vector<std::pair<size_t, size_t>>> chunkRangesPerChannel, 
//                             int* rings, int comm_size,
//                             const half *alpha, const half *beta, const half* m1, const half* m2, half* m1m2, 
//                             ncclComm_t comm, cudaStream_t ncclStream, int M, int N, int K, float& allReduceTime, float& cublasTime,
//                             bool waitOnStreamsAfterFirstChunk, bool singleMatMulForAllChannels, bool perRankChunkMatMulForFirstChunk,
//                             bool combineMatMulForMultipleRanksInFineGrained, int combinedMatMulSize, bool useCustomIndexing, 
//                             int combineMatMulForChannelsInCustomIndexing, int rank) {
//   float elapsedTime = 0;
//   // MPI_Barrier(MPI_COMM_WORLD);
//   //TODO: Make this work for non power of 2 sizes.
//   int* fineGrainedSyncGlobalMem = syncGlobalMem;
//   int* coarseGrainedSyncGlobalMem = syncGlobalMem + nChannels;
//   int channelStreamsToWaitOn[32];
//   int numChannelsToWaitOn = 0;

//   //Process First Chunk of Each Channel and on the same stream write to Global Memory's Sync Location
//   {
//     int chunkIdx = 0;
//     if (useCustomIndexing && combineMatMulForChannelsInCustomIndexing > 1) {
//       for (int channel = 0; channel < chunkRangesPerChannel[chunkIdx].size();) {
//         int userRank = comm_size-1;

//         auto chunkRange = chunkRangesPerChannel[chunkIdx][channel];
//         size_t chunkSize = chunkRange.second - chunkRange.first;
//         size_t perRankChunkSize = chunkSize / comm_size;

//         int startIdx = (rings[channel*comm_size + userRank] * nChannels + channel) * perRankChunkSize;
//         int endIdx = startIdx + perRankChunkSize;//chunkRange.second;
//         int contiguousStartIdx = startIdx;
//         int contiguousEndIdx = endIdx;
//         int combinedChannels;

//         for (combinedChannels = 1; combinedChannels < combineMatMulForChannelsInCustomIndexing; combinedChannels++) {
//           int _startIdx = (rings[(channel + combinedChannels)*comm_size + userRank] * nChannels + channel + combinedChannels) * perRankChunkSize;
//           int _endIdx = _startIdx + perRankChunkSize;//chunkRange.second;

//           if (contiguousStartIdx == _endIdx) {
//             //Can the contiguous region be extended on left side
//             contiguousStartIdx = _startIdx;
//           } else if (contiguousEndIdx == _startIdx) {
//             //Can the contiguous region be extended on right side
//             contiguousEndIdx = _endIdx;
//           } else {
//             //Contiguous region cannot be extended at all
//             break;
//           }
//         }

//         cublasProcessMatMulChunk(handles[channel], streams[channel], channel, chunkIdx, fineGrainedSyncGlobalMem, contiguousStartIdx, contiguousEndIdx, alpha, beta, m1, m2, m1m2, M, N, K);
//         writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[channel]>>>(fineGrainedSyncGlobalMem, 1, channel, channel+ combinedChannels - 1, combinedChannels);
//         channelStreamsToWaitOn[numChannelsToWaitOn] = channel;
//         numChannelsToWaitOn++;
//         channel += combinedChannels;
//       }
//     } else {
//       for (int channel = 0; channel < chunkRangesPerChannel[chunkIdx].size(); channel++) {
//         auto chunkRange = chunkRangesPerChannel[chunkIdx][channel];
//         int userRank = comm_size-1;

//         if (perRankChunkMatMulForFirstChunk) {
//           size_t chunkSize = chunkRange.second - chunkRange.first;
//           size_t perRankChunkSize = chunkSize / comm_size;
//           int startIdx = 0;
//           int endIdx = 0;
//           if (useCustomIndexing) {
//             startIdx = (rings[channel*comm_size + userRank] * nChannels + channel) * perRankChunkSize;
//             endIdx = startIdx + perRankChunkSize;//chunkRange.second;
//           }
//           else 
//           {
//             startIdx = chunkRange.first + rings[channel*comm_size + userRank] * perRankChunkSize;
//             endIdx = startIdx + perRankChunkSize;//chunkRange.second;
//           }
//             // if (iter == 0)
//             // printf("227: endIdx %d startIdx %d rank %d userRank %d peerRank %d\n", endIdx, startIdx, rank, userRank, rings[channel*comm_size + userRank]);
//             cublasProcessMatMulChunk(handles[channel], streams[channel], channel, chunkIdx, fineGrainedSyncGlobalMem, startIdx, endIdx, alpha, beta, m1, m2, m1m2, M, N, K);
//             writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[channel]>>>(fineGrainedSyncGlobalMem, (comm_size-1 - userRank) + 1, channel, channel, 1);
//         } else {
//           cublasProcessMatMulChunk(handles[channel], streams[channel], channel, chunkIdx, coarseGrainedSyncGlobalMem, chunkRange.first, 
//                                   chunkRange.second, alpha, beta, m1, m2, m1m2, M, N, K);
//           writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[channel]>>>(coarseGrainedSyncGlobalMem, chunkIdx, channel, channel, 1);
//         }

//       // CUDACHECK(cudaStreamSynchronize(stream));
//       }
//     }

//     if (!useCustomIndexing && !perRankChunkMatMulForFirstChunk && waitOnStreamsAfterFirstChunk)
//       waitOnStreams(streams, nChannels);
//   }
//   //TODO: Call NCCL on a high priority stream so that it runs first than the next kernels.

//   //Call nccl on NCCL's stream.
//   NCCLCHECK(ncclAllReduce2((const void*)m1, (void*)m2, (void*)m1m2, syncGlobalMem, M*N, M, N, K, ncclHalf, ncclSum, comm, ncclStream));
  
//   if (useCustomIndexing) {
//     for (int c = 0; c < numChannelsToWaitOn; c++) {
//       CUDACHECK(cudaStreamSynchronize(streams[channelStreamsToWaitOn[c]]));
//     }
//   }
//   if (!useCustomIndexing or perRankChunkMatMulForFirstChunk or not waitOnStreamsAfterFirstChunk)
//     waitOnStreams(streams, nChannels);
  
//   if (perRankChunkMatMulForFirstChunk) {
//     int chunkIdx = 0;
//     for (int c = 0; c < nChannels; c++) {
//       userRankForChannels[c] = comm_size - 2;
//     }
    


//     for (int userRank = comm_size - 2; userRank >= 0; userRank--) {
//       if (useCustomIndexing && combineMatMulForChannelsInCustomIndexing > 1) {
//         int channelStreamsToWaitOn[32];
//         int numChannelsToWaitOn = 0;

//         for (int channel = 0; channel < chunkRangesPerChannel[chunkIdx].size();) {
//           auto chunkRange = chunkRangesPerChannel[chunkIdx][channel];
//           size_t chunkSize = chunkRange.second - chunkRange.first;
//           size_t perRankChunkSize = chunkSize / comm_size;

//           int startIdx = (rings[channel*comm_size + userRank] * nChannels + channel) * perRankChunkSize;
//           int endIdx = startIdx + perRankChunkSize;//chunkRange.second;
//           int contiguousStartIdx = startIdx;
//           int contiguousEndIdx = endIdx;
//           int combinedChannels;

//           for (combinedChannels = 1; combinedChannels < combineMatMulForChannelsInCustomIndexing; combinedChannels++) {
//             int _startIdx = (rings[(channel + combinedChannels)*comm_size + userRank] * nChannels + channel + combinedChannels) * perRankChunkSize;
//             int _endIdx = _startIdx + perRankChunkSize;//chunkRange.second;

//             if (contiguousStartIdx == _endIdx) {
//               //Can the contiguous region be extended on left side
//               contiguousStartIdx = _startIdx;
//             } else if (contiguousEndIdx == _startIdx) {
//               //Can the contiguous region be extended on right side
//               contiguousEndIdx = _endIdx;
//             } else {
//               //Contiguous region cannot be extended at all
//               break;
//             }
//           }

//           cublasProcessMatMulChunk(handles[channel], streams[channel], channel, chunkIdx, fineGrainedSyncGlobalMem, contiguousStartIdx, contiguousEndIdx, alpha, beta, m1, m2, m1m2, M, N, K);
//           writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[channel]>>>(fineGrainedSyncGlobalMem, comm_size-1 - userRank + 1, channel, channel+ combinedChannels - 1, combinedChannels);
//           channelStreamsToWaitOn[numChannelsToWaitOn] = channel;
//           numChannelsToWaitOn++;
//           channel += combinedChannels;
//         }

//         for (int c = 0; c < numChannelsToWaitOn; c++) {
//           CUDACHECK(cudaStreamSynchronize(streams[channelStreamsToWaitOn[c]]));
//         }
//         // waitOnStreams(streams, nChannels);

//       } else {
//         for (int channel = 0; channel < chunkRangesPerChannel[chunkIdx].size(); channel++) {
//           int& userRankForChannel = userRankForChannels[channel];
//           if (userRankForChannel < 0) 
//             continue;
//           auto chunkRange = chunkRangesPerChannel[chunkIdx][channel];
//           size_t chunkSize = chunkRange.second - chunkRange.first;
//           size_t perRankChunkSize = chunkSize / comm_size;
//           if (useCustomIndexing) {
//             int startIdx = (rings[channel*comm_size + userRank] * nChannels + channel) * perRankChunkSize;
//             int endIdx = startIdx + perRankChunkSize;

//             cublasProcessMatMulChunk(handles[channel], streams[channel], channel, chunkIdx, fineGrainedSyncGlobalMem, startIdx, endIdx, alpha, beta, m1, m2, m1m2, M, N, K);
//             writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[channel]>>>(fineGrainedSyncGlobalMem, (comm_size-1 - (userRankForChannel)) + 1, channel, channel, 1);
//             userRankForChannel -= 1;

//           } else if (combineMatMulForMultipleRanksInFineGrained) {

//             int startIdx = chunkRange.first + rings[channel*comm_size + userRankForChannel] * perRankChunkSize;
//             // if (iter == 0) if (rank == 3 && channel == 0) {
//             //   printf("startIdx %d rings[channel*comm_size + userRank] * perRankChunkSize; %d\n", startIdx, rings[channel*comm_size + userRank] * perRankChunkSize);
//             // } 
//             int endIdx = startIdx + perRankChunkSize;//chunkRange.second;
//             int contiguousMatMulRank = 1;
//             int contiguousEndIdx = endIdx;
//             int contiguousStartIdx = startIdx;
//             for (contiguousMatMulRank = 1; contiguousMatMulRank < combinedMatMulSize; contiguousMatMulRank++) {
//               //Go through all next ranks and see the contiguous ones.
//               int rankInRing = rings[channel*comm_size + (userRankForChannel - contiguousMatMulRank)] * perRankChunkSize;
//               int _startIdx = chunkRange.first + rankInRing;
//               int _endIdx = _startIdx + perRankChunkSize;//chunkRange.second;
//               // if (iter == 0) if (rank == 3 && channel == 0) {
//               //   printf("startIdx %d _startIdx %d _endIdx %d rankInRing %d\n", startIdx, _startIdx, _endIdx, rankInRing);
//               // } 
//               if (contiguousStartIdx == _endIdx) {
//                 //Can the contiguous region be extended on left side
//                 contiguousStartIdx = _startIdx;
//               } else if (contiguousEndIdx == _startIdx) {
//                 //Can the contiguous region be extended on right side
//                 contiguousEndIdx = _endIdx;
//               } else {
//                 //Contiguous region cannot be extended at all
//                 break;
//               }
//             }
//             // if (iter == 0) if (rank == 3)
//             //   printf("260: channel %d endIdx %d startIdx %d rank %d userRank %d peerRank %d userRankForChannel %d contiguousMatMulRank %d\n", 
//             //         channel, contiguousEndIdx, contiguousStartIdx, rank, userRank, rings[channel*comm_size + userRankForChannel], userRankForChannel, contiguousMatMulRank);
//             cublasProcessMatMulChunk(handles[channel], streams[channel], channel, chunkIdx, fineGrainedSyncGlobalMem, contiguousStartIdx, contiguousEndIdx, alpha, beta, m1, m2, m1m2, M, N, K);
//             writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[channel]>>>(fineGrainedSyncGlobalMem, (comm_size-1 - (userRankForChannel - (contiguousMatMulRank - 1))) + 1, channel, channel, 1);
//             userRankForChannel -= contiguousMatMulRank;
//           }
//         }

//         waitOnStreams(streams, nChannels);
//       }
//     }
//     // CUDACHECK(cudaStreamSynchronize(stream));
//   }

//   //Now there is no need to do matmul chunk by chunk for each channel but 
//   //matmul can be done collectively on chunks of all channels
//   if (singleMatMulForAllChannels) {
//     for (int chunkIdx = 1; chunkIdx < chunkRangesPerChannel.size(); chunkIdx++) {
//       size_t firstChannelChunkStart = chunkRangesPerChannel[chunkIdx][0].first;
//       size_t lastChannelChunkEnd = chunkRangesPerChannel[chunkIdx][nChannels - 1].second;
//       cublasProcessMatMulChunk(handles[0], streams[0], 0, chunkIdx, coarseGrainedSyncGlobalMem, 
//         firstChannelChunkStart, lastChannelChunkEnd, alpha, beta, m1, m2, m1m2, M, N, K);
//       // CUDACHECK(cudaStreamSynchronize(stream));
//       writeProcessedRowsToSyncGlobalMem<<<1, DIVUP(nChannels, 32) * 32, 0, streams[0]>>>(coarseGrainedSyncGlobalMem, chunkIdx, 0, nChannels - 1, nChannels);
//       waitOnStreams(streams, 1);
//     }
//   }
//   else
//   {
//     for (int chunkIdx = 1; chunkIdx < chunkRangesPerChannel.size(); chunkIdx++) {
//       for (int channel = 0; channel < chunkRangesPerChannel[chunkIdx].size(); channel++) {
//         auto chunkRange = chunkRangesPerChannel[chunkIdx][channel];
//         cublasProcessMatMulChunk(handles[channel], streams[channel], channel, chunkIdx, coarseGrainedSyncGlobalMem, chunkRange.first, chunkRange.second, alpha, beta, m1, m2, m1m2, M, N, K);
//         writeProcessedRowsToSyncGlobalMem<<<1, DIVUP(nChannels, 32) * 32, 0, streams[channel]>>>(coarseGrainedSyncGlobalMem, chunkIdx, channel, channel, 1);
//       }

//       waitOnStreams(streams, nChannels);
//     }
//   }

//   // writeProcessedRowsToSyncGlobalMem<<<1, DIVUP(nChannels, 32)*32, 0, stream>>>(syncGlobalMem, M*N, nChannels);
//   // CUDACHECK(cudaStreamSynchronize(stream));
//   // CUDACHECK(cudaDeviceSynchronize());
//   elapsedTime = 0;

//   allReduceTime += elapsedTime;
// }
#endif 

__global__ void writeProcessedRowsToSyncGlobalMem(int* syncGlobalMem, int chunkIdx, int firstChannel, int lastChannel, int nChannels) {
  int threadId = threadIdx.x + blockIdx.x  * blockDim.x;
  if (threadId < nChannels) {
    syncGlobalMem[firstChannel + threadId] = chunkIdx;
  }
}

void cublasProcessMatMulChunk(int rank, cublasHandle_t handle, cudaStream_t stream, int channel, int chunkIdx, 
                              int* syncGlobalMem, size_t chunkStart, size_t chunkEnd, 
                              const half* alpha, const half* beta, const half* m1, const half* m2, half* m1m2, int M, int N, int K)
{
  int rows = DIVUP(chunkEnd - chunkStart, N);
  int startRow = DIVUP(chunkStart, N);
  // if (chunkStart != 0 && chunkEnd % N == 0) {
  //   if (nChunks % 2 == 1) {
  //     startRow += 1;
  //     rows -= 2;
  //   } else {
  //     //startRow -= 1;
  //     rows -= 1;
  //   }
  // }
  if (chunkStart == 0) 
    rows = min(rows, M);
  else {
    if (M < startRow + rows) {
      rows = M - startRow;
    }
  }
  // if ((chunkStart <= 1572864 and chunkEnd >= 1572864) or chunkEnd < 1572864) {
  // if (rank == 1) {
  //   printf("rank %d number %d chunkStart %d chunkEnd %d startRow %d rows %d\n", 
  //           rank, chunkStart/1048576, chunkStart, chunkEnd, startRow, rows);
  // }
  if (rows <= 0 or startRow >= M or rows > M or startRow + rows > M) return;

  CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
  rows, N, K, 
  alpha,
  m1 + startRow * K, CUDA_R_16F, rows,
  m2, CUDA_R_16F, K,
  beta, 
  m1m2 + startRow * N, CUDA_R_16F, rows,
  CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
  
}

void waitOnStreams(cudaStream_t* streams, int nStreams) {
  for (int i = 0; i < nStreams; i++) {
    CUDACHECK(cudaStreamSynchronize(streams[i]));
  }
}


void pipe_overlapped_with_streams(int iter, cublasHandle_t* handles, cudaStream_t* streams, int nStreams, 
                            int comm_size,
                            const half *alpha, const half *beta, const half* m1, const half* m2, half* m1m2, 
                            ncclComm_t comm, cudaStream_t ncclStream, int M, int N, int K,
                            int rank) {
  size_t chunkSize = (M*N) / nStreams;
  //std::cout << "chunkSize " << chunkSize << std::endl;
  cublasProcessMatMulChunk(rank, handles[0], streams[0], 0, 0, 
    nullptr, 0, chunkSize, alpha, beta, m1, m2, m1m2, M, N, K);
  CUDACHECK(cudaStreamSynchronize(streams[0]));

  NCCLCHECK(ncclAllReduce(m1m2 + 0, m1m2 + 0, chunkSize, ncclHalf, ncclSum, comm, ncclStream));

  for (int chunkStart = chunkSize; chunkStart < M*N; chunkStart += chunkSize) {
    cublasProcessMatMulChunk(rank, handles[0], streams[0], 0, 0, 
      nullptr, chunkStart, chunkStart + chunkSize, alpha, beta, m1, m2, m1m2, M, N, K);

    CUDACHECK(cudaStreamSynchronize(streams[0]));
    CUDACHECK(cudaStreamSynchronize(ncclStream));
    if (chunkSize + chunkStart <= M*N)
      NCCLCHECK(ncclAllReduce(m1m2 + chunkStart, m1m2 + chunkStart, chunkSize, ncclHalf, ncclSum, comm, ncclStream));
  }

  CUDACHECK(cudaStreamSynchronize(ncclStream));
}

void pipe_cublas_overlapped(int iter, cublasHandle_t* handles, cudaStream_t* streams, int* syncGlobalMem, int nChannels, 
  std::vector<std::vector<std::pair<size_t, size_t>>> chunkRangesPerChannel, 
                            int* rings, int comm_size,
                            const half *alpha, const half *beta, const half* m1, const half* m2, half* m1m2, 
                            ncclComm_t comm, cudaStream_t ncclStream, int M, int N, int K, float& allReduceTime, float& cublasTime,
                            bool singleMatMulForAllChannels,
                            bool useCustomIndexing, 
                            int combineMatMulForChannelsInCustomIndexing,
                            int maxCombinedRanks,
                            bool combinedRanksInFirstIteration,
                            bool singleMatMulForFirstChunk,
                            int rank) {
  float elapsedTime = 0;
  // MPI_Barrier(MPI_COMM_WORLD);
  //TODO: Make this work for non power of 2 sizes.
  int* fineGrainedSyncGlobalMem = syncGlobalMem;
  int* coarseGrainedSyncGlobalMem = syncGlobalMem + nChannels;
  int channelStreamsToWaitOn[32];
  int numChannelsToWaitOn = 0;
  int userRank = comm_size-1;
  int chunkIdx = 0;

  //Process First Chunk of Each Channel and on the same stream write to Global Memory's Sync Location
  if (singleMatMulForFirstChunk) {
    size_t firstChannelChunkStart = chunkRangesPerChannel[chunkIdx][0].first;
    size_t lastChannelChunkEnd = chunkRangesPerChannel[chunkIdx][nChannels - 1].second;
    cublasProcessMatMulChunk(rank, handles[0], streams[0], 0, chunkIdx, coarseGrainedSyncGlobalMem, 
      firstChannelChunkStart, lastChannelChunkEnd, alpha, beta, m1, m2, m1m2, M, N, K);
    // CUDACHECK(cudaStreamSynchronize(stream));
    writeProcessedRowsToSyncGlobalMem<<<1, DIVUP(nChannels, 32) * 32, 0, streams[0]>>>(fineGrainedSyncGlobalMem, comm_size, 0, nChannels - 1, nChannels);
    waitOnStreams(streams, 1);
    userRank = -1;
  } else {
    if (useCustomIndexing && combineMatMulForChannelsInCustomIndexing >= 1) {
      if (maxCombinedRanks <= 1 || !combinedRanksInFirstIteration) {
        for (int channel = 0; channel < chunkRangesPerChannel[chunkIdx].size();) {

          auto chunkRange = chunkRangesPerChannel[chunkIdx][channel];
          size_t chunkSize = chunkRange.second - chunkRange.first;
          size_t perRankChunkSize = chunkSize / comm_size;

          int startIdx = (rings[channel*comm_size + userRank] * nChannels + channel) * perRankChunkSize;
          int endIdx = startIdx + perRankChunkSize;//chunkRange.second;
          int contiguousStartIdx = startIdx;
          int contiguousEndIdx = endIdx;
          int combinedChannels;

          for (combinedChannels = 1; combinedChannels < min(nChannels, combineMatMulForChannelsInCustomIndexing); combinedChannels++) {
            int _startIdx = (rings[(channel + combinedChannels)*comm_size + userRank] * nChannels + channel + combinedChannels) * perRankChunkSize;
            int _endIdx = _startIdx + perRankChunkSize;//chunkRange.second;

            if (contiguousStartIdx == _endIdx) {
              //Can the contiguous region be extended on left side
              contiguousStartIdx = _startIdx;
            } else if (contiguousEndIdx == _startIdx) {
              //Can the contiguous region be extended on right side
              contiguousEndIdx = _endIdx;
            } else {
              //Contiguous region cannot be extended at all
              break;
            }
          }

          // if (contiguousStartIdx <= 0 && contiguousEndIdx >= 0)
          // printf("contiguousStartIdx %d contiguousEndIdx %d userRank %d rank %d\n", contiguousStartIdx, contiguousEndIdx, userRank, rank);
          cublasProcessMatMulChunk(rank, handles[channel], streams[channel], channel, chunkIdx, 
                                  fineGrainedSyncGlobalMem, contiguousStartIdx, contiguousEndIdx, alpha, beta, m1, m2, m1m2, M, N, K);
          writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[channel]>>>(fineGrainedSyncGlobalMem, 1, channel, channel+ combinedChannels - 1, combinedChannels);
          channelStreamsToWaitOn[numChannelsToWaitOn] = channel;
          numChannelsToWaitOn++;
          channel += combinedChannels;
        }

        userRank -= 1;
      } else if (combinedRanksInFirstIteration) {
        int combinedRanks;
        int contiguousStartIdx = -1;
        int contiguousEndIdx = -1;
        for (combinedRanks = 0; combinedRanks < maxCombinedRanks && userRank - combinedRanks >= 0; combinedRanks++) {
          auto chunkRange = chunkRangesPerChannel[chunkIdx][0];
          size_t chunkSize = chunkRange.second - chunkRange.first;
          size_t perRankChunkSize = chunkSize / comm_size;
          int firstChannelStartIdx = (rings[0*comm_size + (userRank - combinedRanks)] * nChannels + 0) * perRankChunkSize;
          int lastChannelEndIdx = (rings[(nChannels - 1)*comm_size + (userRank - combinedRanks)] * nChannels + nChannels - 1) * perRankChunkSize + perRankChunkSize;

          if (contiguousStartIdx == -1) contiguousStartIdx = firstChannelStartIdx;
          if (contiguousEndIdx == -1) contiguousEndIdx = lastChannelEndIdx;

          if (combinedRanks >= 1) {
            if (contiguousStartIdx == lastChannelEndIdx) {
              //Can the contiguous region be extended on left side
              contiguousStartIdx = firstChannelStartIdx;
            } else if (contiguousEndIdx == firstChannelStartIdx) {
              //Can the contiguous region be extended on right side
              contiguousEndIdx = lastChannelEndIdx;
            } else {
              //Contiguous region cannot be extended
              break;
            }
          }
        }
      //  if(combinedRanksInFirstIteration) std::cout << "combinedRanks " << combinedRanks << std::endl;
        cublasProcessMatMulChunk(rank, handles[0], streams[0], 0, chunkIdx, fineGrainedSyncGlobalMem, contiguousStartIdx, contiguousEndIdx, alpha, beta, m1, m2, m1m2, M, N, K);
        writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[0]>>>(fineGrainedSyncGlobalMem, comm_size-1 - (userRank - (combinedRanks - 1)) + 1, 0, 11, nChannels);
        channelStreamsToWaitOn[numChannelsToWaitOn] = 0;
        numChannelsToWaitOn++;
        userRank -= combinedRanks;
        for (int c = 0; c < numChannelsToWaitOn; c++) {
          CUDACHECK(cudaStreamSynchronize(streams[channelStreamsToWaitOn[c]]));
        }
      }
    }

    // if (!useCustomIndexing  && waitOnStreamsAfterFirstChunk)
    //   waitOnStreams(streams, nChannels);
  }
  //TODO: Call NCCL on a high priority stream so that it runs first than the next kernels.

  //Call nccl on NCCL's stream.
  NCCLCHECK(ncclAllReduceOverlapMatMul((const void*)m1, (void*)m2, (void*)m1m2, syncGlobalMem, M*N, M, N, K, N, 0, ncclHalf, ncclSum, comm, ncclStream));
  
  if (useCustomIndexing) {
    for (int c = 0; c < numChannelsToWaitOn; c++) {
      CUDACHECK(cudaStreamSynchronize(streams[channelStreamsToWaitOn[c]]));
    }
  }
  // if (!useCustomIndexing or not waitOnStreamsAfterFirstChunk)
  //   waitOnStreams(streams, nChannels);
  
  // for (int c = 0; c < nChannels; c++) {
  //   userRankForChannels[c] = comm_size - 2;
  // }
    //std::cout << "userRank " << userRank << std::endl;
  for (; userRank >= 0; ) {
    int combinedRanks = 1;

    if (useCustomIndexing && combineMatMulForChannelsInCustomIndexing >= 1) {
      int channelStreamsToWaitOn[32];
      int numChannelsToWaitOn = 0;
      combinedRanks = 0;

      //for (combinedRanks = 0; combinedRanks < maxCombinedRanks && userRank - combinedRanks >= 0;) {
      int combinedChannels;
      if (maxCombinedRanks <= 1) {
        combinedRanks = 1;

        for (int channel = 0; channel < chunkRangesPerChannel[chunkIdx].size();) {
          auto chunkRange = chunkRangesPerChannel[chunkIdx][channel];
          size_t chunkSize = chunkRange.second - chunkRange.first;
          size_t perRankChunkSize = chunkSize / comm_size;

          int startIdx = (rings[channel*comm_size + (userRank)] * nChannels + channel) * perRankChunkSize;
          int endIdx = startIdx + perRankChunkSize;//chunkRange.second;
          int contiguousStartIdx = -1;
          int contiguousEndIdx = -1;
          if (contiguousStartIdx == -1) contiguousStartIdx = startIdx;
          if (contiguousEndIdx == -1) contiguousEndIdx = endIdx;

          for (combinedChannels = 1; combinedChannels < min(combineMatMulForChannelsInCustomIndexing, nChannels); combinedChannels++) {
            int _startIdx = (rings[(channel + combinedChannels)*comm_size + (userRank)] * nChannels + channel + combinedChannels) * perRankChunkSize;
            int _endIdx = _startIdx + perRankChunkSize;//chunkRange.second;

            if (contiguousStartIdx == _endIdx) {
              //Can the contiguous region be extended on left side
              contiguousStartIdx = _startIdx;
            } else if (contiguousEndIdx == _startIdx) {
              //Can the contiguous region be extended on right side
              contiguousEndIdx = _endIdx;
            } else {
              //Contiguous region cannot be extended
              break;
            }
          }
          // if (contiguousStartIdx <= 0 && contiguousEndIdx >= 0)
          // printf("600: contiguousStartIdx %d contiguousEndIdx %d userRank %d combinedRanks %d rank %d\n", contiguousStartIdx, contiguousEndIdx, userRank, combinedRanks, rank);
          if (combineMatMulForChannelsInCustomIndexing <= nChannels) {
            cublasProcessMatMulChunk(rank, handles[channel], streams[channel], channel, chunkIdx, fineGrainedSyncGlobalMem, contiguousStartIdx, contiguousEndIdx, alpha, beta, m1, m2, m1m2, M, N, K);
            writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[channel]>>>(fineGrainedSyncGlobalMem, comm_size-1 - userRank + 1, channel, channel+ combinedChannels - 1, combinedChannels);
            channelStreamsToWaitOn[numChannelsToWaitOn] = channel;
            numChannelsToWaitOn++;
          }

          channel += combinedChannels;
        }

        for (int c = 0; c < numChannelsToWaitOn; c++) {
          CUDACHECK(cudaStreamSynchronize(streams[channelStreamsToWaitOn[c]]));
        }
      } else {
        int contiguousStartIdx = -1;
        int contiguousEndIdx = -1;
        for (combinedRanks = 0; combinedRanks < maxCombinedRanks && userRank - combinedRanks >= 0; combinedRanks++) {
          auto chunkRange = chunkRangesPerChannel[chunkIdx][0];
          size_t chunkSize = chunkRange.second - chunkRange.first;
          size_t perRankChunkSize = chunkSize / comm_size;
          int firstChannelStartIdx = (rings[0*comm_size + (userRank - combinedRanks)] * nChannels + 0) * perRankChunkSize;
          int lastChannelEndIdx = (rings[(nChannels - 1)*comm_size + (userRank - combinedRanks)] * nChannels + nChannels - 1) * perRankChunkSize + perRankChunkSize;

          if (contiguousStartIdx == -1) contiguousStartIdx = firstChannelStartIdx;
          if (contiguousEndIdx == -1) contiguousEndIdx = lastChannelEndIdx;

          if (combinedRanks >= 1) {
            if (contiguousStartIdx == lastChannelEndIdx) {
              //Can the contiguous region be extended on left side
              contiguousStartIdx = firstChannelStartIdx;
            } else if (contiguousEndIdx == firstChannelStartIdx) {
              //Can the contiguous region be extended on right side
              contiguousEndIdx = lastChannelEndIdx;
            } else {
              //Contiguous region cannot be extended
              break;
            }
          }
        }

        cublasProcessMatMulChunk(rank, handles[0], streams[0], 0, chunkIdx, fineGrainedSyncGlobalMem, contiguousStartIdx, contiguousEndIdx, alpha, beta, m1, m2, m1m2, M, N, K);
        writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[0]>>>(fineGrainedSyncGlobalMem, comm_size-1 - (userRank - (combinedRanks - 1)) + 1, 0, 11, nChannels);
        channelStreamsToWaitOn[numChannelsToWaitOn] = 0;
        numChannelsToWaitOn++;

        for (int c = 0; c < numChannelsToWaitOn; c++) {
          CUDACHECK(cudaStreamSynchronize(streams[channelStreamsToWaitOn[c]]));
        }
      }


      // waitOnStreams(streams, nChannels);
      userRank -= combinedRanks;
    }
    // CUDACHECK(cudaStreamSynchronize(stream));
  }

  //Now there is no need to do matmul chunk by chunk for each channel but 
  //matmul can be done collectively on chunks of all channels
  if (singleMatMulForAllChannels) {
    for (int chunkIdx = 1; chunkIdx < chunkRangesPerChannel.size(); chunkIdx++) {
      size_t firstChannelChunkStart = chunkRangesPerChannel[chunkIdx][0].first;
      size_t lastChannelChunkEnd = chunkRangesPerChannel[chunkIdx][nChannels - 1].second;
      cublasProcessMatMulChunk(rank, handles[0], streams[0], 0, chunkIdx, coarseGrainedSyncGlobalMem, 
        firstChannelChunkStart, lastChannelChunkEnd, alpha, beta, m1, m2, m1m2, M, N, K);
      // CUDACHECK(cudaStreamSynchronize(stream));
      writeProcessedRowsToSyncGlobalMem<<<1, DIVUP(nChannels, 32) * 32, 0, streams[0]>>>(coarseGrainedSyncGlobalMem, chunkIdx, 0, nChannels - 1, nChannels);
      waitOnStreams(streams, 1);
    }
  }
  else
  {
    for (int chunkIdx = 1; chunkIdx < chunkRangesPerChannel.size(); chunkIdx++) {
      for (int channel = 0; channel < chunkRangesPerChannel[chunkIdx].size(); channel++) {
        auto chunkRange = chunkRangesPerChannel[chunkIdx][channel];
        cublasProcessMatMulChunk(rank, handles[channel], streams[channel], channel, chunkIdx, coarseGrainedSyncGlobalMem, chunkRange.first, chunkRange.second, alpha, beta, m1, m2, m1m2, M, N, K);
        writeProcessedRowsToSyncGlobalMem<<<1, DIVUP(nChannels, 32) * 32, 0, streams[channel]>>>(coarseGrainedSyncGlobalMem, chunkIdx, channel, channel, 1);
      }

      waitOnStreams(streams, nChannels);
    }
  }

  // writeProcessedRowsToSyncGlobalMem<<<1, DIVUP(nChannels, 32)*32, 0, stream>>>(syncGlobalMem, M*N, nChannels);
  // CUDACHECK(cudaStreamSynchronize(stream));
  // CUDACHECK(cudaDeviceSynchronize());
  elapsedTime = 0;

  allReduceTime += elapsedTime;
}

bool mpiRef(const float* m1, const float* m2, float* m1m2, int M, int N, int K, int comm_size, int rank = -1)
{
  // printf("Starting Matmul\n");
  // float* expected = new float[M*N];
  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < N; ++j) {
  //     int k = 0;
  //     for (k = 0; k < K; ++k) 
  //     { 
  //           expected[i*N +j] += m1[i*K + k] * m2[k*N + j];
  //     }
  //   }
  // }
  // printf("Starting AllReduce\n");
  // float* allreduceOut = new float[M*N];
  // MPI_Allreduce(expected, allreduceOut, M*N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  // for (size_t i0 = 0; i0 < M*N; i0++) {
  //   if (!eqFloat(allreduceOut[i0], m1m2[i0])) {
  //     printf("Mismatch at %ld : ref '%f', computed '%f'\n",i0, allreduceOut[i0], m1m2[i0]);
  //     return false;
  //   }
  // }

  for (size_t i0 = 0; i0 < M*N; i0++) {
    float ref = K*comm_size;
    if (!eqFloat(ref, m1m2[i0])) {
      printf("rankk %d Mismatch at %ld : ref '%f', computed '%f'\n",rank, i0, ref, m1m2[i0]);
      return false;
    }
  }
  return true;
}

template<typename T>
std::vector<std::vector<std::pair<size_t, size_t>>> getChunkRangesPerChannel(int rank, size_t matrixSize, int nranks) 
{
  std::vector<std::vector<std::pair<size_t, size_t>>> chunkRangesPerChannel; 
  std::vector<std::vector<std::tuple<int, int, int, int>>> chunkBlocks;

  assert (atoi(getenv ("NCCL_MIN_NCHANNELS")) == atoi(getenv ("NCCL_MAX_NCHANNELS")));
  int nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));
  int nThreads = atoi(getenv("NCCL_NTHREADS"));
  // int nThreadsLL128 = atoi(getenv ("NCCL_LL128_NTHREADS"));
  int channelBuffSize = atoi(getenv("NCCL_BUFFSIZE"));

  const int stepSize = channelBuffSize / (sizeof(T)*NCCL_STEPS);
  const size_t chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = nChannels*(ssize_t)chunkSize;

  if (rank == 0)
    printf("loopSize %d chunkSize %d\n", loopSize, chunkSize);
  for (size_t gridOffset = 0; gridOffset < matrixSize; gridOffset += nranks * loopSize) {
    //Push vector for first chunk of each channel
    chunkRangesPerChannel.push_back(std::vector<std::pair<size_t, size_t>>());
    for (int channel = 0; channel < nChannels; channel++) {
      //Push default start and end offset for channel's first chunk
      chunkRangesPerChannel[chunkRangesPerChannel.size() - 1].push_back(std::make_pair(0,0));
    }
  }
  
  for (int channel = 0; channel < nChannels; channel++) {
    int chunkIdx = 0;
    for (size_t gridOffset = 0; gridOffset < matrixSize; gridOffset += nranks * loopSize, chunkIdx++) {
      size_t realChunkSize = min(chunkSize, DIVUP(matrixSize-gridOffset,nranks*nChannels));
      if (matrixSize %3 == 0) {
        ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T) * 3);
      } else {
        ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T));
      }
      ssize_t chunkOffset = gridOffset + channel*nranks*realChunkSize;
      int nelem = min(realChunkSize, matrixSize-chunkOffset);
      if (rank == 0)
        std::cout << "cpu-channel " << channel << " [" << chunkOffset << ", " << (chunkOffset + nranks*realChunkSize) << "]" << std::endl;
      chunkRangesPerChannel[chunkIdx][channel] = std::make_pair(chunkOffset, (chunkOffset + nranks*realChunkSize));
    }
  }

  return chunkRangesPerChannel;
}

template<typename T>
std::vector<std::vector<std::tuple<int, int, int, int>>> getChunkBlocks
  (int rank, size_t matrixSize, int nranks, int* rings, int MATMUL_M, int MATMUL_N,
  const int realChunkCols, int& maxRealChunkRows) 
{
  std::vector<std::vector<std::tuple<int, int, int, int>>> chunkBlocks;

  assert (atoi(getenv ("NCCL_MIN_NCHANNELS")) == atoi(getenv ("NCCL_MAX_NCHANNELS")));
  int nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));
  int nThreads = atoi(getenv("NCCL_NTHREADS"));
  // int nThreadsLL128 = atoi(getenv ("NCCL_LL128_NTHREADS"));
  int channelBuffSize = atoi(getenv("NCCL_BUFFSIZE"));

  const int stepSize = channelBuffSize / (sizeof(T)*NCCL_STEPS);
  const size_t chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
  maxRealChunkRows = 0;

  printf("matrixSize %d chunkSize %d nranks * loopSize %d\n", matrixSize, chunkSize, nranks * loopSize);
  for (int userRank = nranks - 1; userRank >= 0; userRank--) {
    chunkBlocks.push_back(std::vector<std::tuple<int, int, int, int>>());
    int combinedRanks = 1;
    for (int channel = 0; channel < nChannels; channel++) {
      //TODO: following loop only run for once right now.

      for (size_t gridOffset = 0; gridOffset < matrixSize; gridOffset += nranks * loopSize) {
        size_t realChunkSize = min(chunkSize, DIVUP(matrixSize-gridOffset,nranks*nChannels));
        if (matrixSize %3 == 0 && MATMUL_N != 12288) {
          ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T) * 3);
        } else 
        if (matrixSize % 12288 == 0) {
          ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T) * 12);
        }
        else {
          ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T));
        }

        const int realChunkRows = realChunkSize/realChunkCols;
        const int gridOffsetStartRow = gridOffset / MATMUL_N;

        maxRealChunkRows = std::max (maxRealChunkRows, realChunkRows);

        int chunkIdx = rings[channel*nranks + userRank] * nChannels + channel;
        int chunkStartRow = gridOffsetStartRow + chunkIdx / (MATMUL_N / realChunkCols) * realChunkRows;
        int chunkStartCol = chunkIdx % (MATMUL_N / realChunkCols) * realChunkCols;

        int nelem = min(realChunkSize, (matrixSize - (chunkStartRow * MATMUL_N + (MATMUL_M - chunkStartRow) * (MATMUL_N - (MATMUL_N - chunkStartCol)))));
        int chunkRows = min(min(nelem/realChunkCols, realChunkRows), MATMUL_M - chunkStartRow);
        int chunkCols;
        chunkCols = realChunkCols;
        nelem = chunkCols * chunkRows;

        chunkBlocks[chunkBlocks.size() - 1].push_back(std::make_tuple(chunkStartRow, chunkStartCol, chunkRows, chunkCols));
      }
    }
  }

  return chunkBlocks;
}

__device__ uint get_smid(void) {

     uint ret;

     asm("mov.u32 %0, %smid;" : "=r"(ret) );

     return ret;

}

template<int maxTBs>
__global__ void dummyKernel(volatile int* tileStatusMap, int numTiles, int iter)
{
  // if (iter == 0 && threadIdx.x == 0) {
  //   printf("blockIdx.x %d smid %d\n", blockIdx.x, get_smid());
  // }
  if (get_smid() >= maxTBs)
    return;
  if (threadIdx.x == 0) {
    int i = 0;
    while(i < numTiles) {
      while(tileStatusMap[i] != 1);
      i++;
    }
  }

  __syncthreads();
}

#define MAX_CHANNELS 80

int main(int argc, char** argv){
  const int N_GPUS = 16;
  
  MPI_Init(&argc, &argv);  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ncclComm_t comm;
  CUDACHECK(cudaSetDevice(rank % N_GPUS));
  //initializing NCCL
  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, comm_size, id, rank);
  int ringLength;
  int nChannels;
  int* rings = new int[MAX_CHANNELS * comm_size];
  getNCCLRing(&comm, rings, ringLength, nChannels);

  for (int _rank = 0; _rank < comm_size; _rank++) {
    if (_rank != rank) continue;
    std::cout << "rank: " << rank << ":";
    for (int i = 0; i < ringLength; i++) {
      std::cout << rings[i] << "->";
    }
    std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
  }
  

  // std::cout << "ncclChannel buffsize " << comm.channels[0] << std::endl;
  int epochs = 10;
  cudaStream_t stream;
  int leastStreamPriority = 0, highestStreamPriority = 0;
  CUDACHECK(cudaDeviceGetStreamPriorityRange(&leastStreamPriority, &highestStreamPriority));
  printf("highestStreamPriority %d\n", highestStreamPriority);
  cudaStreamCreateWithPriority(&stream, cudaStreamDefault, highestStreamPriority);

  cudaStream_t cutlassStream;
  cudaStreamCreateWithPriority(&cutlassStream, cudaStreamDefault, leastStreamPriority);

  cublasHandle_t handle;
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));
  CUBLASCHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cublasHandle_t handleWithCutlassStream;
  CUBLASCHECK(cublasCreate(&handleWithCutlassStream));
  CUBLASCHECK(cublasSetStream(handleWithCutlassStream, cutlassStream));
  CUBLASCHECK(cublasSetMathMode(handleWithCutlassStream, CUBLAS_TENSOR_OP_MATH));

  half* dAlpha, *dBeta;
  half alpha = __float2half(1.0);
  CUDACHECK(cudaMalloc(&dAlpha, sizeof(half)));
  CUDACHECK(cudaMemcpy(dAlpha, &alpha, sizeof(half), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMalloc(&dBeta, sizeof(half)));
  half beta = __float2half(0);
  CUDACHECK(cudaMemcpy(dBeta, &beta, sizeof(half), cudaMemcpyHostToDevice));
  CUBLASCHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  CUBLASCHECK(cublasSetPointerMode(handleWithCutlassStream, CUBLAS_POINTER_MODE_DEVICE));


  MPI_Barrier(MPI_COMM_WORLD);

  nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));


  // int BATCH_SIZE[] = {8, 8, 8, 8, 8};
  // int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 1024, /*1.2B Model is 1536*/ 2048, /*2.5B Model is 1920*/ 2048, 
  //                            /*4.2B is 2304*/ 2048, /*8.3B is 3072*/ 4096};
  
  #define GPT2_PARAMS
  #ifdef GPT2_PARAMS
    int SEQUENCE_LENGTH = 1024;  
    // int MODEL_PARALLEL_GPUS[] = {1, 2, 4, 8, 16};
    // float MODEL_PARAMS[] = {0.345, 1.2, 2.5, 4.2, 8.3};

    int BATCH_SIZE[] = {8, 16, 32, 64};
    // int BATCH_SIZE[] = {32, 64, 512, 1024, 2048};
    int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 4096, /*1.2B Model is 1536*/ 4096, /*2.5B Model is 1920*/ 4096, 
                              /*4.2B is 2304*/ 4096};
    int HIDDEN_DIMENSIONS_12CHANNELS[] = {3072, /*345M Model*/ 3072, /*1.2B Model is 1536*/ 3072, /*2.5B Model is 1920*/ 3072, 
                                          /*4.2B is 2304*/ 3072};
    int MODEL_PARALLEL_GPUS[] = {16, 16, 16, 16};
    float MODEL_PARAMS[] = {8.3, 8.3, 8.3, 8.3, 8.3};
  #else
    int SEQUENCE_LENGTH = 2048;  
    // int MODEL_PARALLEL_GPUS[] = {1, 2, 4, 8, 16};
    // float MODEL_PARAMS[] = {0.345, 1.2, 2.5, 4.2, 8.3};

    int BATCH_SIZE[] = {1, 2, 4, 6};
    // int BATCH_SIZE[] = {32, 64, 512, 1024, 2048};
    int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 12288, /*1.2B Model is 1536*/ 12288, /*2.5B Model is 1920*/ 12288, 12288};
    int HIDDEN_DIMENSIONS_12CHANNELS[] = {/*345M Model*/ 12288, /*1.2B Model is 1536*/ 12288, /*2.5B Model is 1920*/ 12288, 12288};
    int MODEL_PARALLEL_GPUS[] = {16, 16, 16, 16};
    float MODEL_PARAMS[] = {137, 137, 137, 137};
  #endif

  for (int model = 0; model < sizeof(HIDDEN_DIMENSIONS)/sizeof(HIDDEN_DIMENSIONS[0]); model++) {
    for (int matMulType = 1; matMulType < 2; matMulType++) {

      int M = BATCH_SIZE[model] * SEQUENCE_LENGTH;
      int N = (nChannels%3 == 0) ? HIDDEN_DIMENSIONS_12CHANNELS[model] : HIDDEN_DIMENSIONS[model];
      int K = N/MODEL_PARALLEL_GPUS[model] * ((matMulType == 0) ? 1 : 4);

      if (rank == 0)
        printf("Model Size %.2f B Params , MatMul: [%d, %d] X [%d, %d]\n", MODEL_PARAMS[model], M, K, K, N);
            
      // if (comm_size != MODEL_PARALLEL_GPUS[model])
      //   continue;
      // Inputs
      half* m1;
      CUDACHECK(cudaMalloc(&m1, M*K * sizeof(half)));
      // cudaMemRandInt(m1, M*K);
      memset_value(m1, __float2half(1.0f), M*K);
      half* m2;
      CUDACHECK(cudaMalloc(&m2, K*N * sizeof(half)));
      // cudaMemRandInt(m2, K*N);
      memset_value(m2, __float2half(1.0f), K*N);
      half* m1m2;
      CUDACHECK(cudaMalloc(&m1m2,  M*N* sizeof(half)));
      
      half* _m1m2;
      CUDACHECK(cudaMalloc(&_m1m2,  M*N* sizeof(half)));

       half* __m1m2;
      CUDACHECK(cudaMalloc(&__m1m2,  M*N* sizeof(half)));

      MPI_Barrier(MPI_COMM_WORLD);
      
      float totalTime = 0;
      float cublasTime = 0;
      float allReduceTime = 0;
      float matmulTime = 0;
      
      #define CUBLAS_BASELINE
      #define CUSTOM_BASELINE

      #ifdef CUBLAS_BASELINE
      for(int iter = 0; iter < 110; iter++) {
        if (rank == 0 and iter % 20 == 0)
          printf("iter %d\n", iter);
        cudaEvent_t startpipe, stoppipe;
        float elapsedTimepipe;
        float __allReduceTime = 0.0f, __cublasTime = 0.0f;
        // MPI_Barrier(MPI_COMM_WORLD);

        double t1 = getCurrentTime();
        // if (rank == 0)
        // printf("executiing\n");
        pipe_rowmajorABC(handle, dAlpha, dBeta, m1, m2, m1m2, comm, stream, M, N, K, __allReduceTime, __cublasTime); 

        double t2 = getCurrentTime();
        // if (rank == 0)
        // printf("executiing done\n");
        if (iter >= 10) {
          totalTime += (t2-t1)*1000.0f;
          allReduceTime += __allReduceTime;
          cublasTime += __cublasTime;
        }
        // MPI_Barrier(MPI_COMM_WORLD);
        if (iter == 0) 
        { 
          float *hm1 = new float[M*K];
          float *hm2 = new float[N*K];
          float *hm1m2 = new float[M*N];
          
          cudaMemcpyHalfDevice2FloatHost(hm1, m1, M*K);
          cudaMemcpyHalfDevice2FloatHost(hm2, m2, N*K);
          cudaMemcpyHalfDevice2FloatHost(hm1m2, m1m2, M*N);
          if (rank == 0)
            printf("checking results at iter %d \n", iter);
          if (!mpiRef(hm1, hm2, hm1m2, M, N, K, comm_size))
            assert(false);
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
      printf("AllReduce+cuBLAS: TotalTime %f ms, AllReduceTime %f ms, cuBLAS Time %f ms\n", totalTime, allReduceTime, cublasTime);
      #endif

      memset_value(m1m2, __float2half(0.0f), M*N);

      totalTime = 0.0;
      allReduceTime = 0;
      matmulTime = 0;

      int chunkRows;
      int chunkCols = 512;
      assert(N % chunkCols == 0);
      std::vector<std::vector<std::tuple<int, int, int, int>>> chunkBlocks = getChunkBlocks<half>(rank, M*N, comm_size, rings, M, N, chunkCols, chunkRows) ;

      if (rank == 0 && false) {
        float time = cutlassGeMM(M, N, K, rank, chunkBlocks);

        printf("cutlass GeMM Time: %f\n", time);
      }
      
      
      MPI_Barrier(MPI_COMM_WORLD);

      {
        float cutlassTime = 0.0f;
        allReduceTime = 0.0f;
        //Overlapped AllReduce + CUTLASS
        int length_m = M;
        int length_n = N;
        int length_k = K;
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a((cutlass::half_t*)m1, LayoutInputA::packed(problem_size.mk()));
        cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b((cutlass::half_t*)m2, LayoutInputA::packed(problem_size.kn()));
        cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c((cutlass::half_t*)_m1m2, LayoutInputA::packed(problem_size.mn()));
        cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_d((cutlass::half_t*)m1m2, LayoutInputA::packed(problem_size.mn()));

        // Initialize alpha and beta for dot product computation
        ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
        ElementComputeEpilogue beta = ElementComputeEpilogue(0);

        // Split K dimension into 1 partitions
        int split_k_slices = 1;
                
        //Initialize the memory for thread block to tile map.
        int numTiles = (length_m*length_n)/(ShapeMMAThreadBlock::kMN);
        int* threadBlockToTileMap;
        int* tileIdx;
        int* tileStatusMap;

        CUDACHECK(cudaMalloc(&tileIdx, sizeof(int)));
        CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));

        CUDACHECK(cudaMalloc(&threadBlockToTileMap, numTiles * 2 * sizeof(int)));

        //An array of integers for each tile to indicate if tile is waiting (0) or finished (1)
        CUDACHECK(cudaMalloc(&tileStatusMap, numTiles * 4 * sizeof(int)));
        CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles * 4 * sizeof(int)));

        //Create an array of tile order.
        ShapeMMAThreadBlock shape;
        int *tileOrder = new int[numTiles * 2];

        int idx = 0;
        for (int ty = 0; ty < length_m/ShapeMMAThreadBlock::kM; ty++) {
          for (int tx = 0; tx < length_n/ShapeMMAThreadBlock::kN; tx++) {
            tileOrder[idx] = tx;
            tileOrder[idx + 1] = ty;
            idx += 2;
          } 
        }

        std::vector<int> hChunksForTile;
        int maxChunksForTile = 0;

        const int combinedChunks = nChannels;
        
        if (true) {
          idx = 0;
          int chunk = 0;

          std::set<std::pair<int, int>> chunkTBs;
          std::vector<std::pair<int, int>> tileOrderAsPair;
          std::map<int, std::set<int>> tileToChunks;
              int tilesForChunk = 0;

          for (auto channelChunks: chunkBlocks) {
            for (int channel = 0; channel < channelChunks.size(); channel++) {
              auto chunk = channelChunks[channel];
              int cy = std::get<0>(chunk);
              int cx = std::get<1>(chunk);
              int m = std::get<2>(chunk);
              int n = std::get<3>(chunk);

              int chunkIndex = cy/chunkRows * N/chunkCols + cx/chunkCols;

              //For a chunk get all tiles required to obtain this chunk
              int startTy = (cy/ ShapeMMAThreadBlock::kM) * ShapeMMAThreadBlock::kM;

              for (int ty = startTy; ty < min(cy + m, length_m); ty += ShapeMMAThreadBlock::kM) {
                for (int tx = cx; tx < min(cx + n, length_n); tx += ShapeMMAThreadBlock::kN) {
                  int tileIndex = ty/ShapeMMAThreadBlock::kM * (N/ShapeMMAThreadBlock::kN) + tx/ShapeMMAThreadBlock::kN;
                  if (tileToChunks[tileIndex].count(chunkIndex/combinedChunks) == 0) {
                    tileToChunks[tileIndex].insert(chunkIndex/combinedChunks);
                    // if (rank == 0 && cy >= 7920) {
                    //   printf("cy %d cx %d chunkIndex %d\n", cy, cx, chunkIndex);
                    //   tilesForChunk++;
                    // }
                  }

                  
                  // if (chunkIndex == 0) {
                  //   if (rank == 0) 
                  //     printf("1199: %d x %d -> %d x %d -> %d\n", 
                  //            cy, cx, ty/ShapeMMAThreadBlock::kM, tx/ShapeMMAThreadBlock::kN, tileIndex);
                  // }

                  if (chunkTBs.count(std::make_pair(ty,tx)) == 0) {
                    chunkTBs.insert(std::make_pair(ty,tx));
                    // if (rank == 0 && channel == 0) 
                    //   printf("%d x %d -> %d x %d -> %d\n", cy, cx, ty/ShapeMMAThreadBlock::kM, tx/ShapeMMAThreadBlock::kN, tileIndex);
                    
                    tileOrderAsPair.push_back(std::make_pair(tx/ShapeMMAThreadBlock::kN, ty/ShapeMMAThreadBlock::kM));
                  }
                }
              }

            }
          }

          // if (rank == 0) {
          //   printf("rank %d tilesForChunk %d\n", rank, tilesForChunk);
          // }

          for (auto v : tileToChunks) {
            maxChunksForTile = std::max(maxChunksForTile, (int)v.second.size());
          }

          hChunksForTile = std::vector<int>(maxChunksForTile * numTiles, 0);

          for (auto it : tileToChunks) {
            int i = 0;
            for (int c : it.second) {
              hChunksForTile[it.first * maxChunksForTile + i] = c;
              i++;
            }
            for (; i < maxChunksForTile; i++) {
              hChunksForTile[it.first * maxChunksForTile + i] = -1;
            }
          }

          int _idx = 0;
          for (int i = 0; i < tileOrderAsPair.size(); i++) {
            tileOrder[_idx] = tileOrderAsPair[i].second; //Swap because x ("m") is row and y ("n") is column.
            tileOrder[_idx+1] = tileOrderAsPair[i].first;

            // printf("%d %d\n", tileOrder[_idx], tileOrder[_idx + 1]);
            _idx += 2;
            idx += 2;
          }    
        }

        int* chunksForTile;
        
        CUDACHECK(cudaMemcpy(threadBlockToTileMap, tileOrder, numTiles * 2 * sizeof(int), cudaMemcpyHostToDevice));

        CUDACHECK(cudaMalloc(&chunksForTile, hChunksForTile.size() * sizeof(int)));
        CUDACHECK(cudaMemcpy(chunksForTile, &hChunksForTile[0], hChunksForTile.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        // delete[] tileOrder;

        typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                           tensor_a,  // <- reference to matrix A on device
                                           tensor_b,  // <- reference to matrix B on device
                                           tensor_c,  // <- reference to matrix C on device
                                           tensor_d,  // <- reference to matrix D on device
                                           maxChunksForTile,
                                           chunksForTile,
                                           tileIdx,
                                           threadBlockToTileMap,
                                           tileStatusMap,
                                           {alpha, beta},          // <- tuple of alpha and beta
                                           split_k_slices};        // <- k-dimension split factor

        // Using the arguments, query for extra workspace required for matrix multiplication computation
        size_t workspace_size = Gemm::get_workspace_size(arguments);

        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // Instantiate CUTLASS kernel depending on templates
        Gemm gemm_op;

        // Check the problem size is supported or not 
        cutlass::Status status = gemm_op.can_implement(arguments);
        CUTLASS_CHECK(status);

        status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);
// cudaProfilerStart();
          // CUDACHECK(cudaFuncSetAttribute(dummyKernel<80>,
          //                           cudaFuncAttributeMaxDynamicSharedMemorySize,
          //                           96*1024));
        CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));
        CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles * 4 * sizeof(int)));
        float minSampleTime = 10000000.0f;
        float sampleTime;

        for(int iter = 0; iter < 110; iter++) {
          //CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));

          // CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles * sizeof(int)));

          if (rank == 0 && iter %20 == 0)
            printf("iter %d\n", iter);
          cudaEvent_t startpipe, stoppipe;
          cudaEvent_t cutlassStartPipe, cutlassStopPipe;
          float elapsedTimepipe, cutlassElapsedTimepipe;
          // MPI_Barrier(MPI_COMM_WORLD);

          CUDACHECK(cudaEventCreate(&startpipe));
          CUDACHECK(cudaEventCreate(&stoppipe));
          CUDACHECK(cudaEventCreate(&cutlassStartPipe));
          CUDACHECK(cudaEventCreate(&cutlassStopPipe));
          CUDACHECK(cudaEventRecord(startpipe, stream));
          CUDACHECK(cudaEventRecord(cutlassStartPipe, cutlassStream));

          double t1 = getCurrentTime();         

          //NCCLCHECK(ncclAllReduceMatrix(m1m2, M*N, M, N, N, ncclHalf, ncclSum, comm, stream));
          NCCLCHECK(ncclAllReduceOverlapMatMul((const void*)m1, (void*)m2, (void*)m1m2, tileStatusMap, M*N, M, N, K, chunkCols, iter, ncclHalf, ncclSum, comm, stream));          

          // dummyKernel<80><<<12, 1024, 96*1024, stream>>>(tileStatusMap, numTiles, iter);

           // First run to check results
          status = gemm_op(iter, cutlassStream);
          CUTLASS_CHECK(status);

          // Wait for kernels to finish
          // CUDACHECK(cudaDeviceSynchronize());

          // CUBLASCHECK(cublasGemmEx(handleWithCutlassStream, CUBLAS_OP_N, CUBLAS_OP_N, 
          // N, M, K, 
          // dAlpha,
          // m2, CUDA_R_16F, N,
          // m1, CUDA_R_16F, K,
          // dBeta, 
          // m1m2, CUDA_R_16F, N,
          // CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
          
          // Check processed order of tiles by cutlass.
          // CUDACHECK(cudaDeviceSynchronize());
          // int* hTileProcessedOrder = new int[numTiles*2];
          // CUDACHECK(cudaMemcpy(hTileProcessedOrder, tileStatusMap + numTiles, 2*numTiles*sizeof(int), cudaMemcpyDeviceToHost));
          // if (true) {
          //   for (int i = 0; i < numTiles; i++) {
          //     if (hTileProcessedOrder[2*i] != tileOrder[2*i]) {
          //       printf("1392: hTileProcessedOrder[%d] = %d, tileOder[%d] = %d\n", i, hTileProcessedOrder[2*i], i, tileOrder[2*i]);
                
          //     }
          //     if (hTileProcessedOrder[2*i + 1] != tileOrder[2*i + 1]) {
          //       printf("1396: hTileProcessedOrder[%d] = %d\n", i, hTileProcessedOrder[i]);
          //       break;
          //     }
          //   }
          // } 



          CUDACHECK(cudaEventRecord(cutlassStopPipe, cutlassStream));
          CUDACHECK(cudaEventSynchronize(cutlassStopPipe));
          CUDACHECK(cudaEventElapsedTime(&cutlassElapsedTimepipe, cutlassStartPipe,cutlassStopPipe));
          // printf("cutlassElapsedTimepipe %f\n", cutlassElapsedTimepipe);
          CUDACHECK(cudaEventRecord(stoppipe, stream));
          CUDACHECK(cudaEventSynchronize(stoppipe));
          double t2 = getCurrentTime();

          CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
          CUDACHECK(cudaEventElapsedTime(&cutlassElapsedTimepipe, cutlassStartPipe,cutlassStopPipe));
          
          if (iter >= 10) {
            totalTime += (t2-t1)*1000.0f;
            allReduceTime += elapsedTimepipe;
            cutlassTime += cutlassElapsedTimepipe;
            sampleTime += (t2-t1)*1000.0f;

            if (iter > 10 && iter % 10 == 0) {
              minSampleTime = std::min(minSampleTime, sampleTime*10);
              sampleTime = 0;//(t2-t1)*1000.0f;
            }
          }
          if (iter == 0) 
          { 
            MPI_Barrier(MPI_COMM_WORLD);
            float *hm1 = new float[M*K];
            float *hm2 = new float[N*K];
            float *hm1m2 = new float[M*N];
            
            cudaMemcpyHalfDevice2FloatHost(hm1, m1, M*K);
            cudaMemcpyHalfDevice2FloatHost(hm2, m2, N*K);
            cudaMemcpyHalfDevice2FloatHost(hm1m2, m1m2, M*N);
            if (rank == 0)
              printf("checking results at iter %d %d\n", iter, rank);
            if (!mpiRef(hm1, hm2, hm1m2, M, N, K, comm_size, rank))
              assert(false);
          }
        }
// cudaProfilerStop();

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
          printf("rank %d Overlapped(AllReduce, cutlass) Time: %f ms cutlass: %f ms, allreduceTime: %f ms, minSampleTime: %f ms\n", rank, totalTime, cutlassTime, allReduceTime, minSampleTime);
        
        // printf("rank %d cutlass %f\n", rank, cutlassTime);
      }

      continue;
      /*Matmul time to process N Rows at a time*/ 
      if (rank == 0) {
        totalTime = 0.0f;
        int rowBatch = 1024;

        memset_value(m1m2, __float2half(0.0f), M*N);
        for(int iter = 0; iter < 100; iter++) {
          float elapsedTime = 0.0f;
          cudaEvent_t startpipe, stoppipe;

          CUDACHECK(cudaEventCreate(&startpipe));
          CUDACHECK(cudaEventCreate(&stoppipe));
          CUDACHECK(cudaEventRecord(startpipe, stream));
          for (int row = 0; row < M; row += rowBatch) {
              CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                rowBatch, N, K, 
                dAlpha,
                m1 + row * K, CUDA_R_16F, rowBatch,
                m2, CUDA_R_16F, K,
                dBeta, 
                m1m2 + row * N, CUDA_R_16F, rowBatch,
                CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
          }

          CUDACHECK(cudaEventRecord(stoppipe, stream));

          CUDACHECK(cudaStreamSynchronize(stream));
          CUDACHECK(cudaEventSynchronize(stoppipe));
          CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));
          totalTime += elapsedTime;
        }

        printf("Time to process %d rows per cublas kernel %f\n", rowBatch, totalTime);
      }

      if (rank == 0) {
        totalTime = 0.0f;
        int colBatch = 1024;
        int rowBatch = 2048;

        memset_value(m1m2, __float2half(0.0f), M*N);
        for(int iter = 0; iter < 100; iter++) {
          float elapsedTime = 0.0f;
          cudaEvent_t startpipe, stoppipe;

          CUDACHECK(cudaEventCreate(&startpipe));
          CUDACHECK(cudaEventCreate(&stoppipe));
          CUDACHECK(cudaEventRecord(startpipe, stream));
          for (int row = 0; row < M; row += rowBatch) {
            for (int col = 0; col < N; col += colBatch) {
              CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
                rowBatch, colBatch, K, 
                dAlpha,
                m1 + row * K, CUDA_R_16F, rowBatch,
                m2 + col * K, CUDA_R_16F, colBatch,
                dBeta, 
                m1m2 + row * N, CUDA_R_16F, rowBatch,
                CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
            }
          }

          CUDACHECK(cudaEventRecord(stoppipe, stream));

          CUDACHECK(cudaStreamSynchronize(stream));
          CUDACHECK(cudaEventSynchronize(stoppipe));
          CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));
          if (false && iter == 0) 
          { 
            float *hm1 = new float[M*K];
            float *hm2 = new float[N*K];
            float *hm1m2 = new float[M*N];
            if (rank == 0)
              printf("checking results at iter %d \n", iter);

            cudaMemcpyHalfDevice2FloatHost(hm1, m1, M*K);
            cudaMemcpyHalfDevice2FloatHost(hm2, m2, N*K);
            cudaMemcpyHalfDevice2FloatHost(hm1m2, m1m2, M*N);
            // printf("checking results at iter %d \n", iter);
            if (not mpiRef(hm1, hm2, hm1m2, M, N, K, 1))
              assert(false);
            
            delete hm1;
            delete hm2;
            delete hm1m2;
          }
          totalTime += elapsedTime;
        }

        printf("Time to process %d columns per cublas kernel %f\n", colBatch, totalTime);
      }

      #if 0
      totalTime = 0.0;
      //Now Run the fused+overlapped version
      for(int iter = 0; iter < 100; iter++) {
        if (rank == 0 and iter % 20 == 0)
          printf("iter %d\n", iter);

        cudaEvent_t startpipe, stoppipe;
        float elapsedTimepipe;
        // MPI_Barrier(MPI_COMM_WORLD);

        CUDACHECK(cudaEventCreate(&startpipe));
        CUDACHECK(cudaEventCreate(&stoppipe));
        CUDACHECK(cudaEventRecord(startpipe, stream));
        // if (rank == 0)
        // printf("executiing\n");
        pipe_fuseed(handle, dAlpha, dBeta, m1, m2, m1m2, comm, stream, M, N, K, allReduceTime, cublasTime); 

        CUDACHECK(cudaEventRecord(stoppipe, stream));
        CUDACHECK(cudaEventSynchronize(stoppipe));
        CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
        // if (rank == 0)
        // printf("executiing done\n");
        totalTime += elapsedTimepipe;
        // MPI_Barrier(MPI_COMM_WORLD);
        if (iter == 0) 
        { 
          float *hm1 = new float[M*K];
          float *hm2 = new float[N*K];
          float *hm1m2 = new float[M*N];
          if (rank == 0)
            printf("checking results at iter %d \n", iter);

          cudaMemcpyHalfDevice2FloatHost(hm1, m1, M*K);
          cudaMemcpyHalfDevice2FloatHost(hm2, m2, N*K);
          cudaMemcpyHalfDevice2FloatHost(hm1m2, m1m2, M*N);
          // printf("checking results at iter %d \n", iter);
          assert(mpiRef(hm1, hm2, hm1m2, M, N, K, comm_size));
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
      printf("FusedTotalTime %f ms\n", totalTime);
      #endif

      CUDACHECK(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);

      // memset_value(m1m2, __float2half(0.0f), M*N);

      std::vector<std::vector<std::pair<size_t, size_t>>> chunkRangesPerChannel = getChunkRangesPerChannel<half>(rank, M*N, comm_size);
      if (rank == 0)
        printf("Number of chunks %d\n", chunkRangesPerChannel.size());


      int* syncGlobalMem;
      CUDACHECK(cudaMalloc(&syncGlobalMem, sizeof(int) * nChannels*2));

      cublasHandle_t* handles = new cublasHandle_t[nChannels];
      cudaStream_t* streams = new cudaStream_t[nChannels];;

      for (int c = 0; c < nChannels; c++) {
        CUDACHECK(cudaStreamCreate(&streams[c]));
        CUBLASCHECK(cublasCreate(&handles[c]));
        CUBLASCHECK(cublasSetStream(handles[c], streams[c]));
        CUBLASCHECK(cublasSetMathMode(handles[c], CUBLAS_TENSOR_OP_MATH));
        CUBLASCHECK(cublasSetPointerMode(handles[c], CUBLAS_POINTER_MODE_DEVICE));
      }
      if (false) {
        int M = 16384;
        int N = 4096;
        int K = 256;
        half* m1;
        CUDACHECK(cudaMalloc(&m1, M*K * sizeof(half)));
        // cudaMemRandInt(m1, M*K);
        memset_value(m1, __float2half(1.0f), M*K);
        half* m2;
        CUDACHECK(cudaMalloc(&m2, K*N * sizeof(half)));
        // cudaMemRandInt(m2, K*N);
        memset_value(m2, __float2half(1.0f), K*N);
        half* m1m2;
        CUDACHECK(cudaMalloc(&m1m2,  M*N* sizeof(half)));

        totalTime = 0.0f;
        for(int iter = 0; iter < 120; iter++) {
          int chunkIdx = 0;
          double t1 = getCurrentTime();
          size_t perRankChunkSize = 1024 * N; 
          for (int r = 0; r < 16; r++) {
            size_t start = r * perRankChunkSize;
            size_t end = start + perRankChunkSize;
            // if (iter == 0) {
            //   printf("chunk size: %ld\n", lastChannelChunkEnd - firstChannelChunkStart);
            // }
            cublasProcessMatMulChunk(rank, handles[r], streams[r], r, chunkIdx, syncGlobalMem, 
              start, end, dAlpha, dBeta, m1, m2, m1m2, M, N, K);
            // writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[r]>>>(syncGlobalMem, comm_size-1, 0, 15, 16);
            CUDACHECK(cudaStreamSynchronize(streams[r]));
          }
          
        // CUDACHECK(cudaStreamSynchronize(stream));

          // for (int channel = 0; channel < chunkRangesPerChannel[chunkIdx].size(); channel++) {
            // writeProcessedRowsToSyncGlobalMem<<<1, 32, 0, streams[channel]>>>(syncGlobalMem, chunkIdx, channel, channel, 1);
          // CUDACHECK(cudaStreamSynchronize(stream));
          //}
    
          //waitOnStreams(streams, nChannels);
          CUDACHECK(cudaDeviceSynchronize());

          double t2 = getCurrentTime();
          float elapsedTime2 = (t2 - t1) * 1000.0f;

          if (iter >= 20) {
            totalTime += elapsedTime2;
          }
        }

        std::cout << "Time at 680: " << totalTime << std::endl;

      }

      #if 0
      for (int nStreams = 2; nStreams < 8; nStreams++) {
        if ((M*N) % nStreams != 0) 
          continue;
        memset_value(m1m2, __float2half(0.0f), M*N);

        totalTime = 0.0;
        //Now Run the overlapped(cublas, allreduce) version
        for(int iter = 0; iter < 110; iter++) {
          if (rank == 0 and iter % 20 == 0)
            printf("iter %d\n", iter);

          cudaEvent_t startpipe, stoppipe;
          float elapsedTimepipe;
          // MPI_Barrier(MPI_COMM_WORLD);
          // CUDACHECK(cudaMemset(syncGlobalMem, 0, sizeof(int) * nChannels));
          CUDACHECK(cudaEventCreate(&startpipe));
          CUDACHECK(cudaEventCreate(&stoppipe));
          CUDACHECK(cudaEventRecord(startpipe, stream));
          // if (rank == 0)
          // printf("executiing\n");
          double t1 = getCurrentTime();
          pipe_overlapped_with_streams(iter, handles, streams, nStreams,
                                comm_size, dAlpha, dBeta, m1, m2, m1m2, comm, stream, M, N, K, rank); 

          CUDACHECK(cudaEventRecord(stoppipe, stream));
          CUDACHECK(cudaEventSynchronize(stoppipe));
          CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
          // if (rank == 0)
          // printf("executiing done\n");
          double t2 = getCurrentTime();
          float elapsedTime2 = (t2 - t1) * 1000.0f;

          if (iter >= 10) {
            totalTime += elapsedTimepipe;
            // std::cout << "elapsedTime2 " << elapsedTime2 << " elapsedTimepipe " << elapsedTimepipe << std::endl;
          }
          // MPI_Barrier(MPI_COMM_WORLD);
          if (iter == 0) 
          { 
            float *hm1 = new float[M*K];
            float *hm2 = new float[N*K];
            float *hm1m2 = new float[M*N];
            if (rank == 0)
              printf("checking results at iter %d \n", iter);

            cudaMemcpyHalfDevice2FloatHost(hm1, m1, M*K);
            cudaMemcpyHalfDevice2FloatHost(hm2, m2, N*K);
            cudaMemcpyHalfDevice2FloatHost(hm1m2, m1m2, M*N);
            // printf("checking results at iter %d \n", iter);
            if (not mpiRef(hm1, hm2, hm1m2, M, N, K, comm_size))
              assert(false);
            
            delete hm1;
            delete hm2;
            delete hm1m2;
          }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0)
          printf("StreamsOverlapped(cuBLAS, AllReduce) with {nStreams %d, Time: %f ms}\n", nStreams, totalTime);
        
      }
      #endif

      #if 0
      for (int singleMatMulForAllChannels = 0; singleMatMulForAllChannels < 2; singleMatMulForAllChannels++) {
        if (chunkRangesPerChannel.size() == 1 && singleMatMulForAllChannels == 1)
          continue;
        for (int singleMatMulForFirstChunk = 0; singleMatMulForFirstChunk < 2; singleMatMulForFirstChunk++) {
          if (chunkRangesPerChannel.size() == 1 && singleMatMulForFirstChunk == 1)
            continue;
          for (int contiguousMatMulRanksOrChannels = 1; contiguousMatMulRanksOrChannels <= ((singleMatMulForFirstChunk) ? 1 : nChannels); contiguousMatMulRanksOrChannels += 1) {
            contiguousMatMulRanksOrChannels = max(1, contiguousMatMulRanksOrChannels);
            if (nChannels >= contiguousMatMulRanksOrChannels && nChannels % contiguousMatMulRanksOrChannels != 0) {
              continue;
            }

            if (contiguousMatMulRanksOrChannels> nChannels && contiguousMatMulRanksOrChannels % nChannels != 0) {
              continue;
            }

            for (int combinedMatMulRanks = 1; combinedMatMulRanks <= ((contiguousMatMulRanksOrChannels == nChannels) ? comm_size : 1); combinedMatMulRanks ++) { // (combinedMatMulRanks == 1) ? 2 : combinedMatMulRanks + 2) {
              for (int combineMatMulForFirstStep = 0; combineMatMulForFirstStep < (combinedMatMulRanks > 1 ? 2 : 1); combineMatMulForFirstStep++) {
                memset_value(m1m2, __float2half(0.0f), M*N);
                
                totalTime = 0.0;
                //Now Run the overlapped(cublas, allreduce) version
                for(int iter = 0; iter < 110; iter++) {
                  if (rank == 0 and iter % 20 == 0)
                    printf("iter %d\n", iter);

                  cudaEvent_t startpipe, stoppipe;
                  float elapsedTimepipe;
                  // MPI_Barrier(MPI_COMM_WORLD);
                  // CUDACHECK(cudaMemset(syncGlobalMem, 0, sizeof(int) * nChannels));
                  memset_value(syncGlobalMem, -1, nChannels*2);

                  CUDACHECK(cudaEventCreate(&startpipe));
                  CUDACHECK(cudaEventCreate(&stoppipe));
                  CUDACHECK(cudaEventRecord(startpipe, stream));
                  // if (rank == 0)
                  // printf("executiing\n");
                  double t1 = getCurrentTime();

                  pipe_cublas_overlapped(iter, handles, streams, syncGlobalMem, nChannels, chunkRangesPerChannel, 
                                        rings, comm_size, dAlpha, dBeta, m1, m2, m1m2, comm, stream, M, N, K, allReduceTime, cublasTime, 
                                        (bool)singleMatMulForAllChannels, 
                                        true, contiguousMatMulRanksOrChannels, combinedMatMulRanks, (bool)combineMatMulForFirstStep, 
                                        (bool)singleMatMulForFirstChunk, rank); 

                  CUDACHECK(cudaEventRecord(stoppipe, stream));
                  CUDACHECK(cudaEventSynchronize(stoppipe));
                  CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
                  // if (rank == 0)
                  // printf("executiing done\n");
                  double t2 = getCurrentTime();
                  float elapsedTime2 = (t2 - t1) * 1000.0f;

                  if (iter >= 10) {
                    totalTime += elapsedTimepipe;
                    // std::cout << "elapsedTime2 " << elapsedTime2 << " elapsedTimepipe " << elapsedTimepipe << std::endl;
                  }
                  // MPI_Barrier(MPI_COMM_WORLD);
                  if (iter == 0) 
                  { 
                    float *hm1 = new float[M*K];
                    float *hm2 = new float[N*K];
                    float *hm1m2 = new float[M*N];
                    if (rank == 0)
                      printf("checking results at iter %d \n", iter);

                    cudaMemcpyHalfDevice2FloatHost(hm1, m1, M*K);
                    cudaMemcpyHalfDevice2FloatHost(hm2, m2, N*K);
                    cudaMemcpyHalfDevice2FloatHost(hm1m2, m1m2, M*N);
                    // printf("checking results at iter %d \n", iter);
                    if (not mpiRef(hm1, hm2, hm1m2, M, N, K, comm_size))
                      assert(false);
                    
                    delete hm1;
                    delete hm2;
                    delete hm1m2;
                  }
                }

                MPI_Barrier(MPI_COMM_WORLD);
                if (rank == 0)
                  printf("Overlapped(cuBLAS, AllReduce) with {singleMatMulForAllChannels: %d, contiguousMatMulRanksOrChannels: %d, combinedMatMulRanks %d, combineMatMulForFirstStep %d, singleMatMulForFirstChunk %d, Time: %f ms}\n", 
                  singleMatMulForAllChannels, contiguousMatMulRanksOrChannels, combinedMatMulRanks, combineMatMulForFirstStep, singleMatMulForFirstChunk, totalTime);
              }
            }
          }
        }
      }
  #endif

      CUDACHECK(cudaFree(m1));
      CUDACHECK(cudaFree(m2));
      CUDACHECK(cudaFree(m1m2));
    }
  }


  MPI_Finalize();
}