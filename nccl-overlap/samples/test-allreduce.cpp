#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cstdint>
#include <curand.h>
#include <mpi.h>
#include <stdlib.h>

#define CURANDCHECK(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);            \
  assert(false);}} while(0)

#define CUBLASCHECK(cmd) do {                       \
  cublasStatus_t e = cmd;                           \
  if (e != CUBLAS_STATUS_SUCCESS) {                 \
    printf("Failed: CUBLAS error %s: %d '%d'\n",    \
           __FILE__, __LINE__, cmd);                \
    assert(false);                                  \
  }                                                 \
} while(0)                                          \

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)



// #include "header.h"

float absRelDiff(float u, float v) {
  return abs((u-v)/u);
}
bool eqFloat(float u, float v) {
  if (u == 0.0f || v == 0.0f)
    return u == v;
  return absRelDiff(u, v) <= 1e-5;
}


//Check results of each epoch
template<class T>
bool check_epoch_resuts(const uint64_t size,
                        int rank, int iter,
                        T* d_minibatch_gradients, 
                        T* d_allreduced_gradient)
{
  bool passed = true;
  T *h_minibatch_gradients = (T*)malloc(size * sizeof(T));
  const size_t grad_array_size = size*sizeof(T);

  //Check AllReduced
  CUDACHECK(cudaMemcpy(h_minibatch_gradients, d_minibatch_gradients, 
			  grad_array_size, cudaMemcpyDeviceToHost));

  T *h_reduced_grad = (T*)malloc(grad_array_size);
  CUDACHECK(cudaMemcpy(h_reduced_grad, d_allreduced_gradient, 
                       grad_array_size, cudaMemcpyDeviceToHost));
  T *h_reduced_grad_mpi = (T*)malloc(size * sizeof(T));
  if (sizeof(T) == 4)
    MPI_Allreduce(h_minibatch_gradients, h_reduced_grad_mpi, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  else
    MPI_Allreduce(h_minibatch_gradients, h_reduced_grad_mpi, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (uint64_t i = 0; i < size; i++) {
    if (not eqFloat(h_reduced_grad_mpi[i], h_reduced_grad[i])) {
      printf ("Mismatch in h_reduced_grad at '%d': ref '%f' computed '%f'\n", i, h_reduced_grad_mpi[i], h_reduced_grad[i]);
      passed = false;
      break;
    }
  }
  //Correct these to free
  free(h_minibatch_gradients);
  free(h_reduced_grad_mpi);
  return passed;
}


template<class T>
void traditional_weight_update(const ncclComm_t& comm, const uint64_t size,
                               T* minibatch_gradients, 
                               T* allreduced_gradient, 
                               cudaStream_t& s,
                               ncclDataType_t datatype)
{
  
}

template<class T>
void cudaMemRandInt(T* dst, size_t nelems)
{
  curandGenerator_t gen;
  CURANDCHECK(curandCreateGenerator(&gen,
                                    CURAND_RNG_PSEUDO_DEFAULT));
  CURANDCHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  if (sizeof(T) == sizeof(float))
    CURANDCHECK(curandGenerateUniform(gen, (float*)dst, nelems));
  else
    CURANDCHECK(curandGenerateUniformDouble(gen, (double*)dst, nelems));
  CURANDCHECK(curandDestroyGenerator(gen));
}

// template<class T>
// __global__ void gpu_memset_kernel(T* f, T v, size_t nelems)
// {
//   uint idx = threadIdx.x + blockIdx.x*blockDim.x;
//   if (idx >= nelems)
//     return;
  
//   f[idx] = v;
// }

template<class T>
void memset_value(T*f, T v, size_t nelems) 
{
  T* h_buff = (T*)malloc(sizeof(T)*nelems);

  for (uint64_t i = 0; i < nelems; i++) {
    h_buff[i] = v;
  }

  CUDACHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  free(h_buff);
}
template<class T>
float run(int rank, int64_t size, ncclDataType_t datatype)
{
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ncclComm_t comm;
  CUDACHECK(cudaSetDevice(rank % 16));
  //This code implements the backward pass of a large layer, using model parallelism
  //The whole dataset is divided into several minibatches, with one minibatch assigned to one gpu.
  //The gradients are received from each gpu, then reduced, and weights are updated as:
  //w = w - alpha * g
  
  const int epochs = 1000;

  //allocating and initializing device buffers
  T* minibatch_gradients;
  T* allreduced_gradient;


  CUDACHECK(cudaMalloc(&minibatch_gradients, size * sizeof(T)));
  CUDACHECK(cudaMalloc(&allreduced_gradient, size * sizeof(T)));
  cudaStream_t s;
  
  cudaMemRandInt(minibatch_gradients, size);
  // memset_value(minibatch_gradients, (float)(1<<rank), size);
  CUDACHECK(cudaMemset(allreduced_gradient, 0, size * sizeof(T)));
  //CUDACHECK(cudaMemset(weights, 0, size * sizeof(T)));
  CUDACHECK(cudaStreamCreate(&s));

  //initializing NCCL
  ncclUniqueId id;
  if (rank == 0)
	  ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, comm_size, id, rank);
  MPI_Barrier(MPI_COMM_WORLD);

  // gpu_memset_kernel<<<size/256 + 1,256, 0, s>>>(minibatch_gradients, (T)rank, size);
  int warmup = 1;
  for (int iter = 0; iter < 10; iter++) {
    NCCLCHECK(ncclAllReduce((const void*)minibatch_gradients, 
            (void*)allreduced_gradient, size, datatype, ncclSum, comm, s));

    CUDACHECK(cudaStreamSynchronize(s));
    if (iter == 0)
      assert(check_epoch_resuts(size, rank, iter, minibatch_gradients, allreduced_gradient));
  }

  cudaEvent_t start, stop;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  MPI_Barrier(MPI_COMM_WORLD);
  CUDACHECK(cudaEventRecord(start,0));

  for (int iter = 0; iter < 100; iter++) {
    NCCLCHECK(ncclAllReduce((const void*)minibatch_gradients, 
            (void*)allreduced_gradient, size, datatype, ncclSum, comm, s));

    CUDACHECK(cudaStreamSynchronize(s));
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  CUDACHECK(cudaEventRecord(stop,0));
  CUDACHECK(cudaEventSynchronize(stop));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  //free device buffers
  CUDACHECK(cudaFree(minibatch_gradients));
  CUDACHECK(cudaFree(allreduced_gradient));
  CUDACHECK(cudaStreamDestroy(s));

  //finalizing NCCL
  ncclCommDestroy(comm);

  return elapsedTime;
}

int main(int argc, char* argv[])
{
  //This code implements the backward pass of a large layer, using model parallelism
  //The whole dataset is divided into several minibatches, with one minibatch assigned to one gpu.
  //The gradients are received from each gpu, then reduced, and weights are updated as:
  //w = w - alpha * g
  //Before running this program do "export NCCL_PROTO=LL"

  MPI_Init(&argc, &argv);

  int rank;
  const int size = 8192 * 3072;
  float elapsedTime1 = run<float>(rank, size, ncclFloat);

  printf("Success time: %f\n", elapsedTime1);
  MPI_Finalize();
  return 0;
}
