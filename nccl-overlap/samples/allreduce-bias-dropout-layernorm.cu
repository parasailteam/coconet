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

void pipe(half* input, half* addTensor, half* bias, size_t count, int biasSize, ncclComm_t comm, cudaStream_t stream, float& allReduceTime, float& cublasTime) {
  cudaEvent_t startpipe, stoppipe;
    float elapsedTime = 0;
    // MPI_Barrier(MPI_COMM_WORLD);

    CUDACHECK(cudaEventCreate(&startpipe));
    CUDACHECK(cudaEventCreate(&stoppipe));
    CUDACHECK(cudaEventRecord(startpipe, stream));
    NCCLCHECK(ncclAllReduce(input, input, count, ncclHalf, ncclSum, comm, stream));

    CUDACHECK(cudaEventRecord(stoppipe, stream));

    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaEventSynchronize(stoppipe));
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

    allReduceTime += elapsedTime;

  elapsedTime = 0;
  CUDACHECK(cudaEventCreate(&startpipe));
  CUDACHECK(cudaEventCreate(&stoppipe));
  CUDACHECK(cudaEventRecord(startpipe, stream));
  
  const size_t THREAD_BLOCK_SIZE = 256;
  const int numThreadBlocks = (count % THREAD_BLOCK_SIZE == 0) ? count/THREAD_BLOCK_SIZE : (count / THREAD_BLOCK_SIZE + 1);
  model_update_kernel<<<numThreadBlocks, 256, 0, stream>>>((half*)input, count, (half*)bias, biasSize, (half*)addTensor, count);

  CUDACHECK(cudaEventRecord(stoppipe, stream));
  CUDACHECK(cudaEventSynchronize(stoppipe));
  CUDACHECK(cudaStreamSynchronize(stream));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

  cublasTime  += elapsedTime;

}

void pipe_reducescatter_allgather(half* input, half* addTensor, half* bias, size_t count, int biasSize, ncclComm_t comm, cudaStream_t stream, size_t comm_size, int rank, float& reduceScatterTime, float& allgatherTime, float& cublasTime) {
    cudaEvent_t startpipe, stoppipe;
      float elapsedTime = 0;
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
      model_update_kernel<<<numThreadBlocks, 256, 0, stream>>>((half*)input + per_rank_size * rank, per_rank_size, (half*)bias, biasSize, (half*)addTensor  + per_rank_size * rank, per_rank_size);
      CUDACHECK(cudaEventRecord(stoppipe, stream));

      CUDACHECK(cudaEventSynchronize(stoppipe));
      CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));
      cublasTime += elapsedTime;

  
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
std::vector<std::vector<std::pair<size_t, size_t>>> getChunkRangesPerChannel(size_t matrixSize, int nranks) 
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
      std::cout << "cpu-channel " << channel << " [" << chunkOffset << ", " << (chunkOffset + nranks*realChunkSize) << "]" << std::endl;
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

      for(int iter = 0; iter < 110; iter++) {
        if (rank == 0 and iter % 20 == 0)
          printf("iter %d\n", iter);
        cudaEvent_t startpipe, stoppipe;
        float elapsedTimepipe;
        float __allReduceTime = 0.0f, __cublasTime = 0.0f;
        // MPI_Barrier(MPI_COMM_WORLD);

        CUDACHECK(cudaEventCreate(&startpipe));
        CUDACHECK(cudaEventCreate(&stoppipe));
        CUDACHECK(cudaEventRecord(startpipe, stream));
        // if (rank == 0)
        // printf("executiing\n");
        pipe(m1m2, m1m2, m1m2, M*N, N, comm, stream, __allReduceTime, __cublasTime); 

        CUDACHECK(cudaEventRecord(stoppipe, stream));
        CUDACHECK(cudaEventSynchronize(stoppipe));
        CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
        // if (rank == 0)
        // printf("executiing done\n");
        if (iter >= 10) {
          totalTime += elapsedTimepipe;
          allReduceTime += __allReduceTime;
          cublasTime += __cublasTime;
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
         printf("AllReduce+Dropout-Bias-LayerNorm: TotalTime %f ms, AllReduceTime %f ms, Compute Time %f ms\n", totalTime, allReduceTime, cublasTime);

      totalTime = 0.0;
      allReduceTime = 0;
      matmulTime = 0;

      float computetime = 0, reducescattertime = 0, allgathertime = 0;

      for(int iter = 0; iter < 110; iter++) {
        if (rank == 0 and iter % 20 == 0)
          printf("iter %d\n", iter);
        cudaEvent_t startpipe, stoppipe;
        float elapsedTimepipe;
        float __reducescattertime = 0.0f,__allgathertime=0, __computetime = 0.0f;
        // MPI_Barrier(MPI_COMM_WORLD);

        CUDACHECK(cudaEventCreate(&startpipe));
        CUDACHECK(cudaEventCreate(&stoppipe));
        CUDACHECK(cudaEventRecord(startpipe, stream));
        // if (rank == 0)
        // printf("executiing\n");
        pipe_reducescatter_allgather(m1m2, m1m2, m1m2, M*N, N, comm, stream, comm_size, rank, __reducescattertime, __allgathertime, __computetime); 

        CUDACHECK(cudaEventRecord(stoppipe, stream));
        CUDACHECK(cudaEventSynchronize(stoppipe));
        CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
        // if (rank == 0)
        // printf("executiing done\n");
        if (iter >= 10) {
          totalTime += elapsedTimepipe;
          reducescattertime += __reducescattertime;
          allgathertime += __allgathertime;
          computetime += __computetime;
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
         printf("ReduceScatter+Dropout-Bias-LayerNorm + AllGather: TotalTime %f ms, ReduceScatterTime %f ms, AllGatherTime %f ms, Compute Time %f ms\n", totalTime, reducescattertime, allgathertime, computetime);

      totalTime = 0.0;

      for(int iter = 0; iter < 110; iter++) {
        if (rank == 0 and iter % 20 == 0)
          printf("iter %d\n", iter);
        cudaEvent_t startpipe, stoppipe;
        float elapsedTimepipe;
        float __reducescattertime = 0.0f,__allgathertime=0, __computetime = 0.0f;
        // MPI_Barrier(MPI_COMM_WORLD);

        CUDACHECK(cudaEventCreate(&startpipe));
        CUDACHECK(cudaEventCreate(&stoppipe));
        CUDACHECK(cudaEventRecord(startpipe, stream));
        // if (rank == 0)
        // printf("executiing\n");
        NCCLCHECK(ncclAllReduceDropoutBiasLayernorm(m1m2, m1m2, m1m2, m1m2, M*N, N, ncclHalf, ncclSum, comm, stream));

        CUDACHECK(cudaEventRecord(stoppipe, stream));
        CUDACHECK(cudaEventSynchronize(stoppipe));
        CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
        // if (rank == 0)
        // printf("executiing done\n");
        if (iter >= 10) {
          totalTime += elapsedTimepipe;
        }
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
         printf("Fused AllReduce+Dropout-Bias-LayerNorm: TotalTime %f ms\n", totalTime);


      CUDACHECK(cudaFree(m1));
      CUDACHECK(cudaFree(m2));
      CUDACHECK(cudaFree(m1m2));
    }
  }


  MPI_Finalize();
}