//nvcc matmul-allreduce.cu -std=c++11 -Xcompiler -fopenmp,-O3 -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/mpi -I.. -I/usr/local/cuda/include/ -I ../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c && mpicxx matmul-allreduce.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o matmul-allreduce

#include "header.h"

__global__ void model_update_kernel(__half* input, size_t inputSize, __half* bias, size_t biasSize, __half* add_ten, size_t add_tenSize) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx >= inputSize) {
    return;
  }
  curandState randState;
  curand_init(threadIdx.x, 0, 0, &randState);
  __half in = (curand_uniform(&randState) < 0.1 ? input[idx]  : __float2half(0.0f));
  // input[idx] = __hadd(__hadd(in, bias[idx % biasSize]), add_ten[idx]);
  input[idx] = __hadd_sat(__hadd_sat(in, bias[idx % biasSize]), add_ten[idx]);
}

void pipe(cublasHandle_t handle, const half *alpha, const half *beta, const half* m1, const half* m2, half* m1m2, ncclComm_t comm, cudaStream_t stream, 
          int M, int N, int K, float& allReduceTime, float& cublasTime, float& computeTime) {
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

  NCCLCHECK(ncclAllReduce(m1m2, m1m2, M*N, ncclHalf, ncclSum, comm, stream));

  CUDACHECK(cudaEventRecord(stoppipe, stream));
  CUDACHECK(cudaEventSynchronize(stoppipe));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

  allReduceTime += elapsedTime;

  CUDACHECK(cudaEventCreate(&startpipe));
  CUDACHECK(cudaEventCreate(&stoppipe));
  CUDACHECK(cudaEventRecord(startpipe, stream));
  
  const size_t THREAD_BLOCK_SIZE = 256;
  size_t count = M*N;
  const int numThreadBlocks = (count % THREAD_BLOCK_SIZE == 0) ? count/THREAD_BLOCK_SIZE : (count / THREAD_BLOCK_SIZE + 1);
  model_update_kernel<<<numThreadBlocks, 256, 0, stream>>>((half*)m1m2, M*N, (half*)m1m2, N, (half*)m1m2, M*N);

  CUDACHECK(cudaEventRecord(stoppipe, stream));
  CUDACHECK(cudaEventSynchronize(stoppipe));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

  computeTime  += elapsedTime;

}

void pipe_reducescatter_allgather(cublasHandle_t handle,  const half *alpha, const half *beta, const half* m1, const half* m2, half* m1m2, ncclComm_t comm, 
  cudaStream_t stream,
  int M, int N, int K, size_t comm_size, int rank, float& reduceScatterTime, float& allgatherTime, float& cublasTime, float& computeTime) {
  
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
  
  half* input = m1m2;
  half* addTensor = m1m2;
  half* bias = m1m2;
  int count = M*N;
  elapsedTime = 0;
  int per_rank_size = count / comm_size;
  CUDACHECK(cudaEventCreate(&startpipe));
  CUDACHECK(cudaEventCreate(&stoppipe));
  CUDACHECK(cudaEventRecord(startpipe, stream));
  ncclReduceScatter(input, input + per_rank_size * rank, per_rank_size, ncclHalf, ncclSum, comm, stream);
  CUDACHECK(cudaEventRecord(stoppipe, stream));

  CUDACHECK(cudaEventSynchronize(stoppipe));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));
  reduceScatterTime += elapsedTime;

  elapsedTime = 0;
  CUDACHECK(cudaEventCreate(&startpipe));
  CUDACHECK(cudaEventCreate(&stoppipe));
  CUDACHECK(cudaEventRecord(startpipe, stream));
  const size_t THREAD_BLOCK_SIZE = 256;
  const int numThreadBlocks = (per_rank_size % THREAD_BLOCK_SIZE == 0) ? per_rank_size/THREAD_BLOCK_SIZE : (per_rank_size / THREAD_BLOCK_SIZE + 1);
  model_update_kernel<<<numThreadBlocks, 256, 0, stream>>>((half*)input + per_rank_size * rank, per_rank_size, (half*)bias, N, (half*)addTensor  + per_rank_size * rank, per_rank_size);
  CUDACHECK(cudaEventRecord(stoppipe, stream));

  CUDACHECK(cudaEventSynchronize(stoppipe));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));
  computeTime += elapsedTime;


  elapsedTime = 0;
  CUDACHECK(cudaEventCreate(&startpipe));
  CUDACHECK(cudaEventCreate(&stoppipe));
  CUDACHECK(cudaEventRecord(startpipe, stream));

  ncclAllGather(input + per_rank_size * rank, input, per_rank_size, ncclHalf, comm, stream);

  CUDACHECK(cudaEventRecord(stoppipe, stream));
  CUDACHECK(cudaEventSynchronize(stoppipe));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

  allgatherTime += elapsedTime;

}

int userRankForChannels[80];

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
NCCLCHECK(ncclAllReduceOverlappedMatMulDropoutBiasLayernorm((const void*) m1, (void*)m2, (void*)m1m2, syncGlobalMem, 
    (void*)m1m2, (void*)m1m2, M*N, M, N, K, N,
    ncclHalf, ncclSum, comm, ncclStream));
  
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

bool mpiRef(const float* m1, const float* m2, float* m1m2, int M, int N, int K, int comm_size)
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
      printf("Mismatch at %ld : ref '%f', computed '%f'\n",i0, ref, m1m2[i0]);
      return false;
    }
  }
  return true;
}

template<typename T>
std::vector<std::vector<std::pair<size_t, size_t>>> getChunkRangesPerChannel(size_t matrixSize, int rank, int nranks) 
{
  std::vector<std::vector<std::pair<size_t, size_t>>> chunkRangesPerChannel; 

  assert (atoi(getenv ("NCCL_MIN_NCHANNELS")) == atoi(getenv ("NCCL_MAX_NCHANNELS")));
  int nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));
  int nThreads = atoi(getenv("NCCL_NTHREADS"));
  // int nThreadsLL128 = atoi(getenv ("NCCL_LL128_NTHREADS"));
  int channelBuffSize = atoi(getenv("NCCL_BUFFSIZE"));

  const int stepSize = channelBuffSize / (sizeof(T)*NCCL_STEPS);
  const size_t chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
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
      if (rank == 0) std::cout << "cpu-channel " << channel << " [" << chunkOffset << ", " << (chunkOffset + nranks*realChunkSize) << "]" << std::endl;
      chunkRangesPerChannel[chunkIdx][channel] = std::make_pair(chunkOffset, (chunkOffset + nranks*realChunkSize));
    }
  }

  return chunkRangesPerChannel;
}

#define MAX_CHANNELS 80

int main(int argc, char** argv){
  printf("matmul-allreduce f16\n");
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

  cublasHandle_t handle;
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));
  CUBLASCHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  half* dAlpha, *dBeta;
  half alpha = __float2half(1.0);
  CUDACHECK(cudaMalloc(&dAlpha, sizeof(half)));
  CUDACHECK(cudaMemcpy(dAlpha, &alpha, sizeof(half), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMalloc(&dBeta, sizeof(half)));
  half beta = __float2half(0);
  CUDACHECK(cudaMemcpy(dBeta, &beta, sizeof(half), cudaMemcpyHostToDevice));
  CUBLASCHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));


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

    int BATCH_SIZE[] = {8, 8, 16, 32, 64, 128};
    // int BATCH_SIZE[] = {32, 64, 512, 1024, 2048};
    int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 4096, /*1.2B Model is 1536*/ 4096, /*2.5B Model is 1920*/ 4096, 
                              /*4.2B is 2304*/ 4096, /*8.3B is 3072*/ 4096};
    int HIDDEN_DIMENSIONS_12CHANNELS[] = {3072, /*345M Model*/ 3072, /*1.2B Model is 1536*/ 3072, /*2.5B Model is 1920*/ 3072, 
                                          /*4.2B is 2304*/ 3072, /*8.3B is 3072*/ 3072};
    int MODEL_PARALLEL_GPUS[] = {4, 16, 16, 16, 16, 16};
    float MODEL_PARAMS[] = {8.3, 8.3, 8.3, 8.3, 8.3, 8.3};
  #else
    int SEQUENCE_LENGTH = 2048;  
    // int MODEL_PARALLEL_GPUS[] = {1, 2, 4, 8, 16};
    // float MODEL_PARAMS[] = {0.345, 1.2, 2.5, 4.2, 8.3};

    int BATCH_SIZE[] = {1, 4, 6};
    // int BATCH_SIZE[] = {32, 64, 512, 1024, 2048};
    int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 12288, /*1.2B Model is 1536*/ 12288, /*2.5B Model is 1920*/ 12288};
    int HIDDEN_DIMENSIONS_12CHANNELS[] = {/*345M Model*/ 12288, /*1.2B Model is 1536*/ 12288, /*2.5B Model is 1920*/ 12288};
    int MODEL_PARALLEL_GPUS[] = {4, 8, 8};
    float MODEL_PARAMS[] = {8.3, 8.3, 8.3};
  #endif

  for (int model = 0; model < sizeof(HIDDEN_DIMENSIONS)/sizeof(HIDDEN_DIMENSIONS[0]); model++) {
    for (int matMulType = 0; matMulType < 2; matMulType++) {

      int M = BATCH_SIZE[model] * SEQUENCE_LENGTH;
      int N = (nChannels == 12) ? HIDDEN_DIMENSIONS_12CHANNELS[model] : HIDDEN_DIMENSIONS[model];
      int K = N/MODEL_PARALLEL_GPUS[model] * ((matMulType == 0) ? 1 : 4);

      if (rank == 0)
        printf("Model Size %.2f B Params , MatMul: [%d, %d] X [%d, %d]\n", MODEL_PARAMS[model], M, K, K, N);
            
      if (comm_size != MODEL_PARALLEL_GPUS[model])
        continue;
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
      
      MPI_Barrier(MPI_COMM_WORLD);
      
      float totalTime = 0;
      float cublasTime = 0;
      float allReduceTime = 0;
      float matmulTime = 0;
      float computeTime = 0;

      #define CUBLAS_BASELINE
      #define CUSTOM_BASELINE

      #ifdef CUBLAS_BASELINE
      for(int iter = 0; iter < 110; iter++) {
        if (rank == 0 and iter % 20 == 0)
          printf("iter %d\n", iter);
        cudaEvent_t startpipe, stoppipe;
        float elapsedTimepipe;
        float __allReduceTime = 0.0f, __cublasTime = 0.0f, __computeTime = 0.0f;
        // MPI_Barrier(MPI_COMM_WORLD);

        CUDACHECK(cudaEventCreate(&startpipe));
        CUDACHECK(cudaEventCreate(&stoppipe));
        CUDACHECK(cudaEventRecord(startpipe, stream));
        // if (rank == 0)
        // printf("executiing\n");
        pipe(handle, dAlpha, dBeta, m1, m2, m1m2, comm, stream, M, N, K, __allReduceTime, __cublasTime, __computeTime); 

        CUDACHECK(cudaEventRecord(stoppipe, stream));
        CUDACHECK(cudaEventSynchronize(stoppipe));
        CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
        // if (rank == 0)
        // printf("executiing done\n");
        if (iter >= 10) {
          totalTime += elapsedTimepipe;
          allReduceTime += __allReduceTime;
          cublasTime += __cublasTime;
          computeTime += __computeTime;
        }
        // MPI_Barrier(MPI_COMM_WORLD);
        if (false && iter == 0) 
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
      printf("AllReduce+cuBLAS: TotalTime %f ms, AllReduceTime %f ms, cuBLAS Time %f ms, Dropout-Bias-Layernorm Time %f\n", totalTime, allReduceTime, cublasTime, computeTime);
      #endif

      memset_value(m1m2, __float2half(0.0f), M*N);

      totalTime = 0.0;
      allReduceTime = 0;
      matmulTime = 0;
      /*Custom Matmul time to process N Rows at a time*/ 
      totalTime = 0.0f;
      float computetime = 0, reducescattertime = 0, allgathertime = 0;

      CUDACHECK(cudaDeviceSynchronize());
      MPI_Barrier(MPI_COMM_WORLD);

      for(int iter = 0; iter < 110; iter++) {
        if (rank == 0 and iter % 20 == 0)
          printf("iter %d\n", iter);
        cudaEvent_t startpipe, stoppipe;
        float elapsedTimepipe;
        float __reducescattertime = 0.0f, __allgathertime = 0.0f, __cublasTime = 0.0f, __computeTime = 0.0f;
        // MPI_Barrier(MPI_COMM_WORLD);

        CUDACHECK(cudaEventCreate(&startpipe));
        CUDACHECK(cudaEventCreate(&stoppipe));
        CUDACHECK(cudaEventRecord(startpipe, stream));
        // if (rank == 0)
        // printf("executiing\n");
        
        pipe_reducescatter_allgather(handle, dAlpha, dBeta, m1, m2, m1m2, comm, stream, M, N, K, comm_size, rank, __reducescattertime, __allgathertime, __cublasTime, __computeTime); 

        CUDACHECK(cudaEventRecord(stoppipe, stream));
        CUDACHECK(cudaEventSynchronize(stoppipe));
        CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
        // if (rank == 0)
        // printf("executiing done\n");
        if (iter >= 10) {
          totalTime += elapsedTimepipe;
          reducescattertime += __reducescattertime;
          allgathertime += __allgathertime;
          computetime += __computeTime;
          matmulTime += __cublasTime;
        }
        // MPI_Barrier(MPI_COMM_WORLD);
        if (false && iter == 0) 
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
         printf("ReduceScatter+Dropout-Bias-LayerNorm + AllGather: TotalTime %f ms, ReduceScatterTime %f ms, AllGatherTime %f ms, cuBLAS Time %f ms, Compute Time %f ms\n", 
                totalTime, reducescattertime, allgathertime, matmulTime, computetime);

      
      // memset_value(m1m2, __float2half(0.0f), M*N);
      MPI_Barrier(MPI_COMM_WORLD);

      std::vector<std::vector<std::pair<size_t, size_t>>> chunkRangesPerChannel = getChunkRangesPerChannel<half>(M*N, rank, comm_size);

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

      #if 1
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

            for (int combinedMatMulRanks = 1; combinedMatMulRanks <= ((contiguousMatMulRanksOrChannels == nChannels) ? 4 : 1); combinedMatMulRanks++) {
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
