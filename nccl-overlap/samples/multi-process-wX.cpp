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

template<class T>
void swap(T& x, T&y)
{
  T t = x;
  x = y;
  y = t;
}

//Check results of each epoch
template<class T>
bool check_epoch_resuts(const uint64_t size,
                        int algo, int rank, int iter,
                        T* d_minibatch_gradients, 
                        T* d_allreduced_gradient, 
                        T* d_old_weights, 
                        T* d_weights,
                        T alpha)
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
    if (algo != 2){
      if (h_reduced_grad_mpi[i] != h_reduced_grad[i]) {
        printf ("Mismatch in h_reduced_grad: ref '%f' computed '%f'\n", h_reduced_grad_mpi[i], h_reduced_grad[i]);
	      passed = false;
	      break;
      }
    }
  }

  //Check Weight Update
  T *h_weights = (T*)malloc(size * sizeof(T*));
  T *h_old_weights = (T*)malloc(sizeof(T)*size);

  CUDACHECK(cudaMemcpy(h_old_weights, d_old_weights, 
		       grad_array_size, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_weights, d_weights, 
		       grad_array_size, cudaMemcpyDeviceToHost));


  for (uint64_t i = 0; i < size; i++) {
    T new_weight = h_old_weights[i] + alpha * h_reduced_grad_mpi[i];
    if (new_weight != h_weights[i]) {
      //Lets take a look at the last device only.
      printf("rank %d iter %d h_weights[%ld]-d_weights[%ld] = %f, d_weights[%ld] = %f\n", rank, iter, i, i, new_weight - h_weights[i], i, h_weights[i]);
      passed = false;
    }
    if (!passed)
      break;
  }
  //Correct these to free
  free(h_minibatch_gradients);
  free(h_reduced_grad_mpi);
  free(h_old_weights);
  free(h_weights);

  return passed;
}


template<class T>
void traditional_weight_update(const ncclComm_t& comm, const uint64_t size,
                               T* minibatch_gradients, 
                               T* allreduced_gradient, T* weights,
                               cudaStream_t& s,
                               T alpha, cublasHandle_t& handle,
                               ncclDataType_t datatype)
{
  NCCLCHECK(ncclAllReduce((const void*)minibatch_gradients, 
			  (void*)allreduced_gradient, size,
			  datatype, ncclSum, comm, s));


  //synchronizing on CUDA streams to wait for completion of NCCL operation
  CUDACHECK(cudaStreamSynchronize(s));
  //Update weights on ech gpu
  if (datatype == ncclFloat) {
	  CUBLASCHECK(cublasSaxpy(handle, size, (float*) &alpha, (float*) allreduced_gradient, 1, (float*) weights, 1));
  } else {
	  CUBLASCHECK(cublasDaxpy(handle, size, (double*) &alpha, (double*) allreduced_gradient, 1, (double*) weights, 1));
  }

  CUDACHECK(cudaStreamSynchronize(s));
}


template<class T>
void allreduce_async_weight_update(const ncclComm_t& comm, const uint64_t size,
                                     T* minibatch_gradients, 
                                     T* allreduced_gradient, T* weights,
                                     cudaStream_t& s,
				     T alpha, cublasHandle_t& handle,
				     ncclDataType_t datatype)
{
  NCCLCHECK(ncclAllReduce((const void*)minibatch_gradients, 
			  (void*)allreduced_gradient, size,
			  datatype, ncclSum, comm, s));


  //Update weights on ech gpu
  if (datatype == ncclFloat) {
	  CUBLASCHECK(cublasSaxpy(handle, size, (float*) &alpha, (float*) allreduced_gradient, 1, (float*) weights, 1));
  } else {
	  CUBLASCHECK(cublasDaxpy(handle, size, (double*) &alpha, (double*) allreduced_gradient, 1, (double*) weights, 1));
  }

  CUDACHECK(cudaStreamSynchronize(s));
}


template<class T>
void fused_allreduce_weight_update(const ncclComm_t& comm, const uint64_t size,
                                   T* minibatch_gradients, 
                                  T* allreduced_gradient, T* weights,
                                  cudaStream_t& s,
                                  T* d_alpha, cublasHandle_t& handle,
                                  ncclDataType_t datatype)
{
	NCCLCHECK(ncclAllReduce2((const void*)minibatch_gradients, 
				(void*)allreduced_gradient, weights,
				size,
				d_alpha,
				datatype, ncclSum, comm, s));

  CUDACHECK(cudaStreamSynchronize(s));
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
float run(int algo, int& rank, bool check_results, int64_t size, ncclDataType_t datatype)
{
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ncclComm_t comm;
  CUDACHECK(cudaSetDevice(rank % 8));
  //This code implements the backward pass of a large layer, using model parallelism
  //The whole dataset is divided into several minibatches, with one minibatch assigned to one gpu.
  //The gradients are received from each gpu, then reduced, and weights are updated as:
  //w = w - alpha * g
  
  const int epochs = 1000;

  //allocating and initializing device buffers
  T* minibatch_gradients;
  T* allreduced_gradient;
  T* weights;
  T* d_alpha;
  T* old_weights;

  CUDACHECK(cudaMalloc(&weights, size * sizeof(T)));
  CUDACHECK(cudaMalloc(&minibatch_gradients, size * sizeof(T)));
  CUDACHECK(cudaMalloc(&allreduced_gradient, size * sizeof(T)));
  CUDACHECK(cudaMalloc(&d_alpha, sizeof(T)));
  CUDACHECK(cudaMalloc(&old_weights, size * sizeof(T)));
  cudaStream_t s;
  T alpha = -1.0;
  CUDACHECK(cudaMemcpy(d_alpha, &alpha, sizeof(T), cudaMemcpyHostToDevice));
  cublasHandle_t handle;
  
  cudaMemRandInt(minibatch_gradients, size);
  // memset_value(minibatch_gradients, (float)(1<<rank), size);
  memset_value(weights, (float)1000, size);
  CUDACHECK(cudaMemset(allreduced_gradient, 0, size * sizeof(T)));
  //CUDACHECK(cudaMemset(weights, 0, size * sizeof(T)));
  CUDACHECK(cudaStreamCreate(&s));

  //Parameters for cublas
  CUBLASCHECK(cublasCreate(&handle));

  //initializing NCCL
  ncclUniqueId id;
  if (rank == 0)
	  ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, comm_size, id, rank);
  // gpu_memset_kernel<<<size/256 + 1,256, 0, s>>>(minibatch_gradients, (T)rank, size);
  int warmup = 10;
  for (int iter = 0; iter < warmup; iter++) {
    if (check_results) {
      CUDACHECK(cudaMemcpy(old_weights, weights, size*sizeof(T), 
                           cudaMemcpyDeviceToDevice));
    }
    switch (algo)
    {
      case 0:
        traditional_weight_update(comm, size, minibatch_gradients, allreduced_gradient, weights,
                                  s, alpha, handle, datatype);
        break;
      case 1:
        allreduce_async_weight_update(comm, size, minibatch_gradients, allreduced_gradient, weights,
                                      s, alpha, handle, datatype);
          break;
      case 2:
        fused_allreduce_weight_update(comm, size, minibatch_gradients, allreduced_gradient, weights,
                                      s, d_alpha, handle, datatype);
        break;
      
        default:
          printf("Invalid algo %d", algo);
          break;
    }
    if (check_results)
      assert(check_epoch_resuts(size, algo, rank, iter, minibatch_gradients, allreduced_gradient, old_weights,
                        weights, alpha));
    
    //Swap weights and gradients
    //swap(weights, new_weights);
    //swap(allreduced_gradient, minibatch_gradients);
  }

  cudaEvent_t start, stop;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  MPI_Barrier(MPI_COMM_WORLD);
  CUDACHECK(cudaEventRecord(start,0));

  for (int iter = 0; iter < epochs; iter++) {
    //printf ("iter %d\n", iter);
    switch (algo)
    {
      case 0:
        traditional_weight_update(comm, size, minibatch_gradients, allreduced_gradient, weights,
                                  s, alpha, handle, datatype);
        break;
      case 1:
        allreduce_async_weight_update(comm, size, minibatch_gradients, allreduced_gradient, weights,
                                      s, alpha, handle, datatype);
          break;
      case 2:
        fused_allreduce_weight_update(comm, size, minibatch_gradients, allreduced_gradient, weights,
                                      s, d_alpha, handle, datatype);
        break;
      
        default:
          printf("Invalid algo %d", algo);
          break;
    }
    
    //Swap weights and gradients
    //swap(weights, new_weights);
    swap(allreduced_gradient, minibatch_gradients);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  CUDACHECK(cudaEventRecord(stop,0));
  CUDACHECK(cudaEventSynchronize(stop));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  //free device buffers
  CUDACHECK(cudaFree(minibatch_gradients));
  CUDACHECK(cudaFree(allreduced_gradient));
  CUDACHECK(cudaFree(weights));
  CUDACHECK(cudaStreamDestroy(s));
  CUBLASCHECK(cublasDestroy(handle));

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

  if (argc < 4) {
    printf ("Provide three arguments: algo check_results size perf_eval\n");
    return 0;
  }
  int rank;
  const int algo = atoi(argv[1]);  
  const int check_results = atoi(argv[2]);
  const int size = atoi(argv[3]);
  const int eval = argc >= 5 ? atoi(argv[4]) : 0;
  printf("Using algo %d, check_results %d size %d\n", algo, check_results, size);

  if (eval == 0) {
    float elapsedTime1 = run<float>(algo, rank, (bool)check_results, size, ncclFloat);
    printf("elapsedTime %f\n", elapsedTime1);
  } else {
    printf("%-10s %-10s %-10s %-10s\n","Size","Baseline(ms)","Fused(ms)","Speedup");
    for (uint64_t size = 128; size <= 1024*1024*1024; size = size * 4) {
      float elapsedTime1 = 0;   
      elapsedTime1 = run<float>(0, rank, (bool)check_results, size, ncclFloat);
      float elapsedTime2 = 0.0;
      elapsedTime2 = run<float>(2, rank, (bool)check_results, size, ncclFloat);
      if(rank == 0) {
        printf("%-15.2ld", size);   
        printf("%-15.2f", elapsedTime1);
        printf("%-15.2f", elapsedTime2);
        printf("%-15.2f\n", elapsedTime1/elapsedTime2);
      }
    }
  }

  printf("Success \n");
  MPI_Finalize();
  return 0;
}
