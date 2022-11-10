//nvcc test.cu -I.. -I/usr/local/cuda/include/ -I../../nccl-2/build/include/ -L../../nccl-2/build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -c -lcurand -gencode=arch=compute_70,code=sm_70 &&  mpicxx test.o -I/usr/local/cuda/include/ -I../build/include/ -L../../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -o a.out -Wall -lcurand
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
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <curand_kernel.h>
#include <curand.h>

using namespace cooperative_groups;
#ifndef __HEADER_H__
#define __HEADER_H__

#define DIVUP(x,y) ((x) + (y) - 1)/(y)

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

float absRelDiff(float u, float v) {
  return abs((u-v)/u);
}
bool eqFloat(float u, float v) {
  if (u == 0.0f || v == 0.0f)
    return u == v;
  return absRelDiff(u, v) <= 1e-3;
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

  h[id] = __half2float(f[id]);
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
  CURANDCHECK(curandDestroyGenerator(gen));

  floatToHalfArrayKernel<<<nelems/256+1, 256>>>(dst, tmp, nelems);
  CUDACHECK(cudaFree(tmp));
}

template<class T>
__global__ void memset_kernel(T* f, T v, size_t numElems) {
  size_t id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < numElems)
    f[id] = v;
}

template<class T>
void memset_value(T*f, T v, size_t nelems) 
{
  memset_kernel<<<nelems/256 + 1, 256>>> (f, v, nelems);
  CUDACHECK(cudaDeviceSynchronize());
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

#endif