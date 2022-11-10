#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cstdint>
#include <curand.h>
#include <mpi.h>
#include <string>
#include <vector>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <curand_kernel.h>

#define MAX(x,y) (x < y) ? y : x)

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



#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))

#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);

#define WARP_SIZE 32
#define MAXCHANNELS 32
#define NCCL_MAX_NTHREADS 512
#define NCCL_LL_MAX_NTHREADS NCCL_MAX_NTHREADS
#define NCCL_LL_LINES_PER_THREAD 8
#define NCCL_LL_SLICE_LINES (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS)
#define NCCL_LL_BUFF_LINES (NCCL_LL_SLICE_LINES*NCCL_STEPS)
#define NCCL_LL_BUFF_SIZE (NCCL_LL_BUFF_LINES*sizeof(union ncclLLFifoLine))

#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)

#define NCCL_LL128_MAX_NTHREADS 640
#define NCCL_LL128_ELEMS_PER_THREAD 120

// Receiving from up to 3 sources is more compute intensive than sending
// to 3 dests. Use 70% for reduce and 30% for bcast.
#define NCCL_LL128_SPLIT(nt) ((nt*7/(10*32))*32)

#define NCCL_LL128_SLICE_ELEMS (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)
#define NCCL_LL128_BUFF_ELEMS (NCCL_LL128_SLICE_ELEMS*NCCL_STEPS)
#define NCCL_LL128_BUFF_SIZE (NCCL_LL128_BUFF_ELEMS*sizeof(uint64_t))

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)

#define NCCL_STEPS 8
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1

float absRelDiff(float u, float v) {
  return abs((u-v)/u);
}
bool eqFloat(float u, float v) {
  if (u == 0.0f || v == 0.0f)
    return u == v;
  return absRelDiff(u, v) <= 1e-5;
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

__global__ void floatToHalfArrayKernel(half* h, float* f, size_t num)
{
  int id = threadIdx.x  + blockDim.x * blockIdx.x;
  if (id >= num) return;

  h[id] = __float2half(f[id]);
}
//nvcc test.cu -I.. -I/usr/local/cuda/include/ -I../../nccl-2/build/include/ -L../../nccl-2/build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -c -lcurand -gencode=arch=compute_70,code=sm_70 &&  mpicxx test.o -I/usr/local/cuda/include/ -I../build/include/ -L../../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -o a.out -Wall -lcurand
template<>
void cudaMemRandInt<half>(half* dst, size_t nelems)
{
  float* tmp;
  CUDACHECK(cudaMalloc(&tmp, sizeof(float)*nelems));

  curandGenerator_t gen;
  CURANDCHECK(curandCreateGenerator(&gen,
                                    CURAND_RNG_PSEUDO_DEFAULT));
  CURANDCHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  CURANDCHECK(curandGenerateUniform(gen, (float*)tmp, nelems));
  CUDACHECK(cudaDeviceSynchronize());
  CURANDCHECK(curandDestroyGenerator(gen));
  CUDACHECK(cudaDeviceSynchronize());

  floatToHalfArrayKernel<<<nelems/256+1, 256>>>(dst, tmp, nelems);
  CUDACHECK(cudaDeviceSynchronize());
  CUDACHECK(cudaFree(tmp));
}

template<class T>
void memset_value(T*f, T v, size_t nelems) 
{
  T* h_buff = (T*)malloc(sizeof(T)*nelems);
  assert(h_buff != nullptr);
  for (uint64_t i = 0; i < nelems; i++) {
    h_buff[i] = v;
  }

  CUDACHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  free(h_buff);
}

template<class T>
void memset_identity_values(T*f, size_t nelems) 
{
  T* h_buff = (T*)malloc(sizeof(T)*nelems);

  for (uint64_t i = 0; i < nelems; i++) {
    h_buff[i] = i;
  }

  CUDACHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  free(h_buff);
}

void halfToFloatArray(float* f, half* h, size_t num) 
{
  for (size_t i = 0; i < num; i++) {
    f[i] = __half2float(h[i]);
  }
}

void floatToHalfArray(half* h, float* f, size_t num) 
{
  for (size_t i = 0; i < num; i++) {
    h[i] = __float2half(f[i]);
  }
}

void cudaMemcpyHalfDevice2FloatHost(float* hostFloatArray, half* deviceHalfArray, size_t nelems)
{
  half* tmp = new half[nelems];
  CUDACHECK(cudaMemcpy(tmp, deviceHalfArray, nelems*sizeof(half), cudaMemcpyDeviceToHost));

  halfToFloatArray(hostFloatArray, tmp, nelems);

  delete tmp;
}

double convertTimeValToDouble (struct timeval _time)
{
  return ((double)_time.tv_sec) + ((double)_time.tv_usec)/1000000.0f;
}

struct timeval getTimeOfDay ()
{
  struct timeval _time;

  if (gettimeofday (&_time, NULL) == -1) {
    fprintf (stderr, "gettimeofday returned -1\n");
    perror ("");
    abort ();
  }

  return _time;
}


double getCurrentTime() {
  return convertTimeValToDouble(getTimeOfDay ());
}


