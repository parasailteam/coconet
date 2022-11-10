#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cstdint>
//#include <curand.h>
#include <chrono>
#include <iostream>

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
bool eq_float(T f1, T f2)
{
  return (abs((f1-f2)/f1) <= 1e-4);
}


template<class T>
void swap(T& x, T&y)
{
  T t = x;
  x = y;
  y = t;
}

//Check results of each epoch
template<class T>
bool check_epoch_resuts(int algo, int epoch, const int nDev, const int devs[],
                        const uint64_t num_weight_elems,
                        T** d_minibatch_gradients, 
                        T** d_allreduced_gradient, 
                        T** d_weights,
                        T** d_new_weights, 
                        T** d_alphas, 
                        T** d_betas)
{
  bool passed = true;
  T **h_minibatch_gradients = (T**)malloc(nDev * sizeof(T*));
  const size_t grad_array_size = num_weight_elems*sizeof(T);

  //Check AllReduced
  for (int dev = 0; dev < nDev; dev++) {
    CUDACHECK(cudaSetDevice(dev));
    h_minibatch_gradients[dev] = (T*)malloc(num_weight_elems*sizeof(T));
    CUDACHECK(cudaMemcpy(h_minibatch_gradients[dev], d_minibatch_gradients[dev], 
                         grad_array_size, cudaMemcpyDeviceToHost));
  }

  T *h_reduced_grad = (T*)malloc(grad_array_size);
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaMemcpy(h_reduced_grad, d_allreduced_gradient[0], 
                       grad_array_size, cudaMemcpyDeviceToHost));

  for (uint64_t i = 0; i < num_weight_elems; i++) {
    T sum = 0.0;

    for (int dev = 0; dev < nDev; dev++) {
      sum += h_minibatch_gradients[dev][i];
    }

    if (algo != 2) {
      if (abs((sum - h_reduced_grad[i])/sum) > 1e-5) {
        printf ("Mismatch in h_reduced_grad: ref '%f' computed '%f'\n", sum, h_reduced_grad[i]);
        passed = false;
        break;
      }
    } else {
      h_reduced_grad[i] = sum;
    }
  }

  //Check Weight Update
  T alpha;
  T beta;
  T **h_weights = (T**)malloc(nDev * sizeof(T*));
  T **h_all_reduced_grads = (T**)malloc(nDev * sizeof(T*));

  //Check AllReduced
  for (int dev = 0; dev < nDev; dev++) {
    CUDACHECK(cudaSetDevice(dev));
    h_weights[dev] = (T*)malloc(num_weight_elems*sizeof(T));
    CUDACHECK(cudaMemcpy(h_weights[dev], d_weights[dev], 
                         grad_array_size, cudaMemcpyDeviceToHost));
    if (algo == 2) {
      h_all_reduced_grads[dev] = (T*)malloc(num_weight_elems*sizeof(T));
      CUDACHECK(cudaMemcpy(h_all_reduced_grads[dev], d_allreduced_gradient[dev], 
                            grad_array_size, cudaMemcpyDeviceToHost));
    }
  }

  T **h_new_weights = (T**)malloc(sizeof(T*)*nDev);

  for (int i = 0; i < nDev; i++) {
    T *ptr = (T*)malloc(grad_array_size);
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(ptr, d_new_weights[i], 
                         grad_array_size, cudaMemcpyDeviceToHost));
    h_new_weights[i] = ptr;
  }

  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaMemcpy(&alpha, d_alphas[0], sizeof(T), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(&beta, d_betas[0], sizeof(T), cudaMemcpyDeviceToHost));
  
  for (uint64_t i = 0; i < num_weight_elems; i++) {
    for (int dev = 0; dev < nDev; dev++) 
    //int dev = (i/2048) % nDev;
    {
      T new_weight = alpha * h_weights[dev][i] + beta *  h_reduced_grad[i];
      if (abs((new_weight - h_new_weights[dev][i])/new_weight) > 1e-4) {
        //Lets take a look at the last device only.
        printf("Epoch %d Mismatch in h_new_weights for device %d at [%ld]: ref '%f' computed '%f'\n", epoch, dev, i, new_weight, h_new_weights[dev][i]);
        printf("h_weights[%ld] = %f , h_reduced_gradients[%ld] = %f, h_all_reduced_gradients[%ld] = %f\n", i, h_weights[dev][i], i, h_reduced_grad[i], i, h_all_reduced_grads[dev][i]);
        for (int dev2 = 0; dev2 < nDev; dev2++) {
          printf("%d: %f, ", dev2, h_minibatch_gradients[dev2][i]);
        }
        printf("\n");
        passed = false;
      }
    }
    if (!passed)
      break;
  }

  //Correct these to free
  for (int i = 0; i < nDev; i++) {
    free(h_minibatch_gradients[i]);
    free(h_weights[i]);
    free(h_new_weights[i]);
  }

  free(h_new_weights);
  free(h_reduced_grad);
  free(h_weights);
  free(h_minibatch_gradients);

  return passed;
}

template<class T>
void traditional_weight_update(const int nDev, const int devs[],
                               const ncclComm_t comms[],
                               const uint64_t num_weight_elems,
                               T** minibatch_gradients, 
                               T** allreduced_gradient, T** weights,
                               cudaStream_t* s,
                               T** alphas, T** betas, 
                               cublasHandle_t* handle,
                               ncclDataType_t datatype,
                               double& weight_update_time,
                               double& reduce_time)
{
  auto start = std::chrono::steady_clock::now();

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce((const void*)minibatch_gradients[i], 
                            (void*)allreduced_gradient[i], num_weight_elems,
                            datatype, ncclSum, comms[i], s[i]));
  }

  NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  auto stop = std::chrono::steady_clock::now();

  reduce_time += (stop - start).count()/1e6;

  start = std::chrono::steady_clock::now();
  //Update weights on ech gpu
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    if (datatype == ncclFloat) {
      CUBLASCHECK(cublasSgeam(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, num_weight_elems,
                              1,
                              (float*)alphas[i], (float*)weights[i], num_weight_elems, (float*)betas[i], 
                              (float*)allreduced_gradient[i], num_weight_elems,
                              (float*)weights[i], num_weight_elems));
    } else {
      CUBLASCHECK(cublasDgeam(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, num_weight_elems,
                        1,
                        (double*)alphas[i], (double*)weights[i], num_weight_elems, (double*)betas[i], 
                        (double*)allreduced_gradient[i], num_weight_elems,
                        (double*)weights[i], num_weight_elems));
    }
  }

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  stop = std::chrono::steady_clock::now();
  weight_update_time += (stop - start).count()/1e6;

  // std::cout<<"Weight Update time " << (weight_update_time).count()/1000.0 << " ms" << " All Reduce Time " << reduce_time.count()/1000.0 << std::endl;
  // std::cout<<"Fraction of Weight Update time " << (weight_update_time).count()/(double)((weight_update_time).count()+reduce_time.count()) << std::endl;
}

template<class T>
void allreduce_async_weight_update(const int nDev, const int devs[],
                               const ncclComm_t comms[],
                               const uint64_t num_weight_elems,
                               T** minibatch_gradients, 
                               T** allreduced_gradient, T** weights,
                               cudaStream_t* s,
                               T** alphas, T** betas, 
                               cublasHandle_t* handle,
                               ncclDataType_t datatype)
{
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce((const void*)minibatch_gradients[i], 
                            (void*)allreduced_gradient[i], num_weight_elems,
                            datatype, ncclSum, comms[i], s[i]));
  }

  NCCLCHECK(ncclGroupEnd());

  // //synchronizing on CUDA streams to wait for completion of NCCL operation
  // for (int i = 0; i < nDev; ++i) {
  //   CUDACHECK(cudaSetDevice(i));
  //   CUDACHECK(cudaStreamSynchronize(s[i]));
  // }

  //Update weights on ech gpu
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    if (datatype == ncclFloat) {
      CUBLASCHECK(cublasSaxpy(handle[i], num_weight_elems,
                              (float*)betas[i], 
                              (float*)allreduced_gradient[i], 1,
                              (float*)weights[i], 1));
    } else {
      CUBLASCHECK(cublasDgeam(handle[i], CUBLAS_OP_N, CUBLAS_OP_N, num_weight_elems,
                        1,
                        (double*)alphas[i], (double*)weights[i], num_weight_elems, (double*)betas[i], 
                        (double*)allreduced_gradient[i], num_weight_elems,
                        (double*)weights[i], num_weight_elems));
    }
  }

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }
}

template<class T>
void fused_allreduce_weight_update(const int nDev, const int devs[],
                                   const ncclComm_t comms[],
                                   const uint64_t num_weight_elems,
                                   T** minibatch_gradients, 
                                   T** allreduced_gradient, T** weights,
                                   cudaStream_t* s,
                                   T** gradient_factors, 
                                   cublasHandle_t* handle,
                                   ncclDataType_t datatype)
{
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduce2((const void*)minibatch_gradients[i], 
                            (void*)allreduced_gradient[i], 
                            weights[i],
                            num_weight_elems,
                            gradient_factors[i],
                            datatype, ncclSum, comms[i], s[i]));
  }

  NCCLCHECK(ncclGroupEnd());
  
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }
}

template<class T>
bool check_epoch_results_scattered(int algo, int epoch, const int nDev, const int devs[],
                                  T** d_minibatch_gradients, 
                                  T*** h_weights, T*** d_new_weights, 
                                  size_t nbuffs, const size_t *h_buffSizes)
                        
{
  bool passed = true;
  T ***h_minibatch_gradients = (T***)malloc(nDev * sizeof(T**));
  // const size_t grad_array_size = num_weight_elems*sizeof(T);

  //Check AllReduced
  for (int dev = 0; dev < nDev; dev++) {
    CUDACHECK(cudaSetDevice(dev));
    size_t s = 0;
    h_minibatch_gradients[dev] = (T**)malloc(nbuffs*sizeof(T**));
    for (size_t buff = 0; buff < nbuffs; buff++) {
      h_minibatch_gradients[dev][buff] = (T*)malloc(h_buffSizes[buff]*sizeof(T));
      CUDACHECK(cudaMemcpy(h_minibatch_gradients[dev][buff], d_minibatch_gradients[dev]+s, 
                           h_buffSizes[buff]*sizeof(T), cudaMemcpyDeviceToHost));
      s += h_buffSizes[buff];
      }
  }

  T **h_reduced_grad = (T**)malloc(sizeof(T*)*nbuffs);

  // CUDACHECK(cudaSetDevice(0));
  // CUDACHECK(cudaMemcpy(h_reduced_grad, d_allreduced_gradient[0], 
  //                      grad_array_size, cudaMemcpyDeviceToHost));

  for (size_t buff = 0; buff < nbuffs; buff++) {
    h_reduced_grad[buff] = (T*)malloc(h_buffSizes[buff]*sizeof(T));
    for (uint64_t i = 0; i < h_buffSizes[buff]; i++) {
      T sum = 0.0;

      for (int dev = 0; dev < nDev; dev++) {
        sum += h_minibatch_gradients[dev][buff][i];
      }

      h_reduced_grad[buff][i] = sum;

      if(buff == 1 && i == 0) {
        printf("h_reduced_grad[buff][i] %f\n", h_reduced_grad[buff][i]);
      }
    }
  }

  //Check Weight Update
  T ***h_new_weights = (T***)malloc(nDev * sizeof(T**));

  //Check AllReduced
  for (int dev = 0; dev < nDev; dev++) {
    CUDACHECK(cudaSetDevice(dev));
    h_new_weights[dev] = (T**)malloc(nbuffs*sizeof(T**));
    for (size_t buff = 0; buff < nbuffs; buff++) {
      h_new_weights[dev][buff] = (T*)malloc(h_buffSizes[buff]*sizeof(T));

      CUDACHECK(cudaMemcpy(h_new_weights[dev][buff], d_new_weights[dev][buff], 
                           h_buffSizes[buff]*sizeof(T), cudaMemcpyDeviceToHost));
    }
  }

  for (size_t buff = 0; buff < nbuffs; buff++) {
    for (uint64_t i = 0; i < h_buffSizes[buff]; i++) {
      for (int dev = 0; dev < nDev; dev++) {
        T new_weight = h_weights[dev][buff][i] + 1 *  h_reduced_grad[buff][i];
        if (!eq_float(new_weight, h_new_weights[dev][buff][i]) > 1e-4) {
          printf("Epoch %d Mismatch in h_new_weights for device %d in buff %ld at [%ld]: ref '%f' computed '%f'\n", epoch, dev, buff, i, new_weight, h_new_weights[dev][buff][i]);
          printf("h_weights[%ld] = %f , h_reduced_gradients[%ld] = %f\n", i, h_weights[dev][buff][i], i, h_reduced_grad[buff][i]);
          // for (int dev2 = 0; dev2 < nDev; dev2++) {
          //   printf("%d: %f, ", dev2, h_minibatch_gradients[dev2][i]);
          // }
          // printf("\n");
          passed = false;
        }
      }
      if (!passed)
        break;
    }
  }

  //Correct these to free
  //Remove frees, lets hope the system has significant RAM.

  // for (int i = 0; i < nDev; i++) {
  //   free(h_minibatch_gradients[i]);
  //   free(h_weights[i]);
  //   free(h_new_weights[i]);
  // }

  // free(h_new_weights);
  // free(h_reduced_grad);
  // free(h_weights);
  // free(h_minibatch_gradients);

  return passed;
}

template<class T>
void cudaMemRandInt(T* dst, size_t nelems)
{
  // if (sizeof(T) == sizeof(float))
  //   CURANDCHECK(curandGenerateUniform(gen, (float*)dst, nelems));
  // else
  //   CURANDCHECK(curandGenerateUniformDouble(gen, (double*)dst, nelems));
  // CUDACHECK(cudaDeviceSynchronize());
  // CURANDCHECK(curandDestroyGenerator(gen));
  T* ptr = (T*)malloc(nelems*sizeof(T));
  for(size_t i = 0; i < nelems; i++) {
    ptr[i] = static_cast <T> (rand()) / static_cast <T> (RAND_MAX);
  }

  CUDACHECK(cudaMemcpy(dst, ptr, nelems*sizeof(T), cudaMemcpyHostToDevice));
  CUDACHECK(cudaDeviceSynchronize());
  free(ptr);
}

template<class T>
float run(int algo, bool check_results, uint64_t size, T gradient_factor, 
          ncclDataType_t datatype, double& weight_update_time, double& reduce_time)
{
  ncclComm_t comms[4];
  //This code implements the backward pass of a large layer, using model parallelism
  //The whole dataset is divided into several minibatches, with one minibatch assigned to one gpu.
  //The gradients are received from each gpu, then reduced, and weights are updated as:
  //w = w - alpha * g
  
  //managing 4 devices
  const int nDev = 8;
  int devs[nDev] = { 0, 1, 2, 3, 4, 5, 6, 7 };
  const uint64_t num_weight_elems = size;
  const int epochs = 10;

  //allocating and initializing device buffers
  T** minibatch_gradients = (T**)malloc(nDev * sizeof(T*));
  T** allreduced_gradient  = (T**)malloc(nDev * sizeof(T*));
  T** weights = (T**)malloc(nDev * sizeof(T*));
  T** old_weights = (T**)malloc(nDev * sizeof(T*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  T** alphas = (T**)malloc(nDev * sizeof(T*));
  T** betas = (T**)malloc(nDev * sizeof(T*));
  cublasHandle_t* handle = (cublasHandle_t*)malloc(nDev * sizeof(cublasHandle_t));
  T* h_weights = (T*)malloc(num_weight_elems*sizeof(T));

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(weights + i, num_weight_elems * sizeof(T)));
    CUDACHECK(cudaMalloc(old_weights + i, num_weight_elems * sizeof(T)));
    CUDACHECK(cudaMalloc(minibatch_gradients + i, num_weight_elems * sizeof(T)));
    CUDACHECK(cudaMalloc(allreduced_gradient + i, num_weight_elems * sizeof(T)));
    // printf ("dev %d allreduced_gradient %p weights %p\n", i, allreduced_gradient[i], weights[i]);
    cudaMemRandInt(minibatch_gradients[i], num_weight_elems);
    CUDACHECK(cudaMemset(allreduced_gradient[i], 0, num_weight_elems * sizeof(T)));
    if (i == 0) {
      cudaMemRandInt( weights[i], num_weight_elems);
      CUDACHECK(cudaMemcpy(h_weights, weights[i], num_weight_elems * sizeof(T), cudaMemcpyDeviceToHost));
    } else {
      CUDACHECK(cudaMemcpy(weights[i], h_weights, num_weight_elems * sizeof(T), cudaMemcpyHostToDevice));
    }
    CUDACHECK(cudaStreamCreate(s+i));

    //Parameters for cublas
    CUBLASCHECK(cublasCreate(handle + i));
    T alpha = 1.0;
    CUDACHECK(cudaMalloc(alphas + i, sizeof(T)));
    CUDACHECK(cudaMemcpy(alphas[i], &alpha, sizeof(T), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMalloc(betas + i, sizeof(T)));
    T beta = -gradient_factor;
    CUDACHECK(cudaMemcpy(betas[i], &beta, sizeof(T), cudaMemcpyHostToDevice));
    CUBLASCHECK(cublasSetPointerMode(handle[i], CUBLAS_POINTER_MODE_DEVICE));
    CUBLASCHECK(cublasSetStream(handle[i], s[i]));
  }

  // //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  cudaEvent_t start[nDev], stop[nDev];
  float elapsedTime = 0.0;
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaEventCreate(&start[i]));
    CUDACHECK(cudaEventCreate(&stop[i]));
    CUDACHECK(cudaEventRecord(start[i],s[i]));
  }

  for (int iter = 0; iter < epochs; iter++) {
    if (check_results) {
      for (int dev = 0; dev < nDev; dev++) {
        CUDACHECK(cudaSetDevice(dev));
        CUDACHECK(cudaMemcpy(old_weights[dev], weights[dev], num_weight_elems*sizeof(T), 
                            cudaMemcpyDeviceToDevice));
      }
    }

    switch (algo)
    {
      case 0:
        traditional_weight_update(nDev, devs, comms, num_weight_elems,
                                  minibatch_gradients, allreduced_gradient, weights,
                                  s, alphas, betas, handle,
                                  datatype, weight_update_time, reduce_time);
        break;
      case 1:
        allreduce_async_weight_update(nDev, devs, comms, num_weight_elems,
                                      minibatch_gradients, allreduced_gradient, weights,
                                      s, alphas, betas, handle,
                                      datatype);
          break;
      case 2:
        fused_allreduce_weight_update(nDev, devs, comms, num_weight_elems,
                                      minibatch_gradients, allreduced_gradient, weights,
                                      s, betas, handle,
                                      datatype);
        break;
      
        default:
          printf("Invalid algo %d", algo);
          break;
    }
    
    if (check_results)
      assert(check_epoch_resuts(algo, iter, nDev, devs, num_weight_elems,
                        minibatch_gradients, allreduced_gradient, old_weights,
                        weights, alphas, betas));
    
    //Swap gradients
    swap(allreduced_gradient, minibatch_gradients);
  }
  
  for (int i = 0; i < nDev; i++) {
    float t;
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaEventRecord(stop[i],s[i]));
    CUDACHECK(cudaEventSynchronize(stop[i]));
    CUDACHECK(cudaEventElapsedTime(&t, start[i], stop[i]));
    elapsedTime = max(elapsedTime, t);
  }

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(minibatch_gradients[i]));
    CUDACHECK(cudaFree(allreduced_gradient[i]));
    CUDACHECK(cudaFree(weights[i]));
    CUDACHECK(cudaFree(alphas[i]));
    CUDACHECK(cudaFree(betas[i]));
    CUBLASCHECK(cublasDestroy(handle[i]));
    CUDACHECK(cudaEventDestroy(stop[i]));
    CUDACHECK(cudaEventDestroy(start[i]));
    CUDACHECK(cudaStreamDestroy(s[i]));
  }

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);
  
  return elapsedTime;
}

template<class T>
void fused_allreduce_adam(const int nDev, const int devs[],
                          const int epoch,
                          const ncclComm_t comms[],
                          const uint64_t num_weight_elems,
                          T** minibatch_gradients, 
                          T** weights, T** first_moments, T** second_moments,
                          cudaStream_t* s,
                          T** stepsizes, T** beta1s, T** beta2s,
                          cublasHandle_t* handle,
                          ncclDataType_t datatype)
{
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduceAdam((const void*)minibatch_gradients[i], 
                            nullptr, 
                            weights[i],
                            num_weight_elems,
                            stepsizes[i], first_moments[i], second_moments[i], beta1s[i], beta2s[i],
                            epoch, datatype, ncclSum, comms[i], s[i]));
  }

  NCCLCHECK(ncclGroupEnd());
  
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }
}

//Check results of each epoch
template<class T>
bool check_epoch_results_adam(int algo, int epoch, const int nDev, const int devs[],
                        const uint64_t num_weight_elems,
                        T** d_minibatch_gradients, 
                        T** d_allreduced_gradient, 
                        T** d_weights, T** d_new_weights, 
                        T* cpu_moment, T** d_moments,
                        T* cpu_second_moment, T** d_second_moments,
                        T** d_beta1s, T** d_beta2s, T** d_stepsizes,
                        T** d_epsilons)
{
  bool passed = true;
  T **h_minibatch_gradients = (T**)malloc(nDev * sizeof(T*));
  const size_t grad_array_size = num_weight_elems*sizeof(T);

  //Check AllReduced
  for (int dev = 0; dev < nDev; dev++) {
    CUDACHECK(cudaSetDevice(dev));
    h_minibatch_gradients[dev] = (T*)malloc(num_weight_elems*sizeof(T));
    CUDACHECK(cudaMemcpy(h_minibatch_gradients[dev], d_minibatch_gradients[dev], 
                         grad_array_size, cudaMemcpyDeviceToHost));
  }

  T *h_reduced_grad = (T*)malloc(grad_array_size);
  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaMemcpy(h_reduced_grad, d_allreduced_gradient[0], 
                       grad_array_size, cudaMemcpyDeviceToHost));

  for (uint64_t i = 0; i < num_weight_elems; i++) {
    T sum = 0.0;

    for (int dev = 0; dev < nDev; dev++) {
      sum += h_minibatch_gradients[dev][i];
    }

    if (algo != 2) {
      if (abs((sum - h_reduced_grad[i])/sum) > 1e-5) {
        printf ("Mismatch in h_reduced_grad: ref '%f' computed '%f'\n", sum, h_reduced_grad[i]);
        passed = false;
        break;
      }
    } else {
      h_reduced_grad[i] = sum;
    }
  }

  //Check Weight Update
  T beta1;
  T beta2;
  T stepsize;
  T epsilon;

  T **h_weights = (T**)malloc(nDev * sizeof(T*));
  T **h_all_reduced_grads = (T**)malloc(nDev * sizeof(T*));
  T **h_moments = (T**)malloc(nDev * sizeof(T*));
  T **h_second_moments = (T**)malloc(nDev * sizeof(T*));
  T **h_old_moments = (T**)malloc(nDev * sizeof(T*));
  T **h_old_second_moments = (T**)malloc(nDev * sizeof(T*));

  //Check AllReduced
  for (int dev = 0; dev < nDev; dev++) {
    CUDACHECK(cudaSetDevice(dev));
    h_weights[dev] = (T*)malloc(grad_array_size);
    CUDACHECK(cudaMemcpy(h_weights[dev], d_weights[dev], 
                         grad_array_size, cudaMemcpyDeviceToHost));
    h_moments[dev] = (T*)malloc(grad_array_size);
    CUDACHECK(cudaMemcpy(h_moments[dev], d_moments[dev], 
                         grad_array_size, cudaMemcpyDeviceToHost));
    h_second_moments[dev] = (T*)malloc(grad_array_size);
    CUDACHECK(cudaMemcpy(h_second_moments[dev], d_second_moments[dev], 
                         grad_array_size, cudaMemcpyDeviceToHost));
    if (algo == 2) {
      h_all_reduced_grads[dev] = (T*)malloc(grad_array_size);
      CUDACHECK(cudaMemcpy(h_all_reduced_grads[dev], d_allreduced_gradient[dev], 
                            grad_array_size, cudaMemcpyDeviceToHost));
    }
  }

  T **h_new_weights = (T**)malloc(sizeof(T*)*nDev);

  for (int i = 0; i < nDev; i++) {
    T *ptr = (T*)malloc(grad_array_size);
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(ptr, d_new_weights[i], 
                         grad_array_size, cudaMemcpyDeviceToHost));
    h_new_weights[i] = ptr;
  }

  CUDACHECK(cudaSetDevice(0));
  CUDACHECK(cudaMemcpy(&beta1, d_beta1s[0], sizeof(T), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(&beta2, d_beta2s[0], sizeof(T), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(&stepsize, d_stepsizes[0], sizeof(T), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(&epsilon, d_epsilons[0], sizeof(T), cudaMemcpyDeviceToHost));

  if (epoch == 1) {
    for (uint64_t i = 0; i < num_weight_elems; i++) {
      T old_m = h_old_moments[0][i];
      if (old_m != 0.0) {
        printf("Epoch %d value of h_old_moments %f at index %ld\n", epoch, old_m, i);
        break;
      }
    }
  }

  for (uint64_t i = 0; i < num_weight_elems; i++) {
    T m, v;
    for (int dev = 0; dev < nDev; dev++) {
      T old_m = cpu_moment[i];
      m = beta1 * old_m + (1-beta1) * h_reduced_grad[i];
      if (false && !eq_float(m, h_moments[dev][i])) {
        printf("Epoch %d Mismatch in h_moments for device %d at [%ld]: ref '%f' computed '%f'\n", epoch, dev, i, m, h_moments[dev][i]);
        passed = false;
        break;
      }

      T old_v = cpu_second_moment[i];
      v = beta2 * old_v + (1-beta2) * h_reduced_grad[i]*h_reduced_grad[i];
      if (false && !eq_float(v, h_second_moments[dev][i])) {
        printf("Epoch %d Mismatch in h_second_moments for device %d at [%ld]: ref '%f' computed '%f'\n", epoch, dev, i, v, h_second_moments[dev][i]);
        passed = false;
        break;
      }
      
      
      T m_ = m/(1 - pow(beta1, epoch + 1));
      T v_ = v/(1 - pow(beta2, epoch + 1));
      T x = stepsize * m_ / (sqrt(v_) + epsilon);
      T new_weight = h_weights[dev][i] + x;

      if (!eq_float(new_weight, h_new_weights[dev][i])) {
        //Lets take a look at the last device only.
        printf("Epoch %d Mismatch in h_new_weights for device %d at [%ld]: ref '%f' computed '%f'\n", epoch, dev, i, new_weight, h_new_weights[dev][i]);
        printf("h_weights[%ld] = %f , h_reduced_gradients[%ld] = %f, new_m = %f, new_v = %f, m_ = %f, v_ = %f, old_m = %f, old_v = %f d_new_m = %f\n", i, h_weights[dev][i], i, h_reduced_grad[i], m, v, m_, v_, old_m, old_v, h_moments[dev][i]);
        for (int dev2 = 0; dev2 < nDev; dev2++) {
          printf("%d: %f, ", dev2, h_minibatch_gradients[dev2][i]);
        }
        printf("\n");
        passed = false;
        break;
      }
    }

    cpu_second_moment[i] = v;
    cpu_moment[i] = m;

    if (!passed)
      break;
  }

  //Correct these to free
  for (int i = 0; i < nDev; i++) {
    free(h_minibatch_gradients[i]);
    free(h_weights[i]);
    free(h_new_weights[i]);
  }

  free(h_new_weights);
  free(h_reduced_grad);
  free(h_weights);
  free(h_minibatch_gradients);

  return passed;
}

template<class T>
float run_adam(int algo, bool check_results, uint64_t size, T stepsize, 
          ncclDataType_t datatype, double& weight_update_time, double& reduce_time)
{
  ncclComm_t comms[4];
  //This code implements the backward pass of a large layer, using model parallelism
  //The whole dataset is divided into several minibatches, with one minibatch assigned to one gpu.
  //The gradients are received from each gpu, then reduced, and weights are updated as:
  //w = w - alpha * g
  
  //managing 4 devices
  const int nDev = 4;
  int devs[4] = { 0, 1, 2, 3 };
  const uint64_t num_weight_elems = size;
  const int epochs = 100;

  //allocating and initializing device buffers
  T** minibatch_gradients = (T**)malloc(nDev * sizeof(T*));
  T** allreduced_gradient  = (T**)malloc(nDev * sizeof(T*));
  T** weights = (T**)malloc(nDev * sizeof(T*));
  T** old_weights = (T**)malloc(nDev * sizeof(T*));
  T** moments = (T**)malloc(nDev * sizeof(T));
  T** second_moments = (T**)malloc(nDev * sizeof(T));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  T** beta1s = (T**)malloc(nDev * sizeof(T*));
  T** beta2s = (T**)malloc(nDev * sizeof(T*));
  T** stepsizes = (T**)malloc(nDev * sizeof(T*));
  T** epsilons = (T**)malloc(nDev * sizeof(T*));
  cublasHandle_t* handle = (cublasHandle_t*)malloc(nDev * sizeof(cublasHandle_t));
  T* h_weights = (T*)malloc(num_weight_elems*sizeof(T));
  T* h_moments = (T*)malloc(num_weight_elems*sizeof(T));
  T* h_second_moments = (T*)malloc(num_weight_elems*sizeof(T));

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(weights + i, num_weight_elems * sizeof(T)));
    CUDACHECK(cudaMalloc(old_weights + i, num_weight_elems * sizeof(T)));
    CUDACHECK(cudaMalloc(minibatch_gradients + i, num_weight_elems * sizeof(T)));
    CUDACHECK(cudaMalloc(allreduced_gradient + i, num_weight_elems * sizeof(T)));
    CUDACHECK(cudaMalloc(moments + i, num_weight_elems * sizeof(T)));
    CUDACHECK(cudaMalloc(second_moments + i, num_weight_elems * sizeof(T)));
    // printf ("dev %d allreduced_gradient %p weights %p moments %p second_moments %p\n", i, allreduced_gradient[i], weights[i], moments[i], second_moments[i]);
    cudaMemRandInt(minibatch_gradients[i], num_weight_elems);
    CUDACHECK(cudaMemset(allreduced_gradient[i], 0, num_weight_elems * sizeof(T)));
    if (i == 0) {
      cudaMemRandInt( weights[i], num_weight_elems);
      CUDACHECK(cudaMemcpy(h_weights, weights[i], num_weight_elems * sizeof(T), cudaMemcpyDeviceToHost));
      CUDACHECK(cudaMemset(moments[i], 0, num_weight_elems * sizeof(T)));
      CUDACHECK(cudaMemset(second_moments[i], 0, num_weight_elems * sizeof(T)));
      CUDACHECK(cudaMemcpy(h_moments, moments[i], num_weight_elems * sizeof(T), cudaMemcpyDeviceToHost));
      CUDACHECK(cudaMemcpy(h_second_moments, second_moments[i], num_weight_elems * sizeof(T), cudaMemcpyDeviceToHost));
    } else {
      CUDACHECK(cudaMemcpy(weights[i], h_weights, num_weight_elems * sizeof(T), cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(second_moments[i], h_second_moments, num_weight_elems * sizeof(T), cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemcpy(moments[i], h_moments, num_weight_elems * sizeof(T), cudaMemcpyHostToDevice));
    }
    CUDACHECK(cudaStreamCreate(s+i));

    //Parameters for cublas
    CUBLASCHECK(cublasCreate(handle + i));
    T beta1 = 0.5;
    CUDACHECK(cudaMalloc(beta1s + i, sizeof(T)));
    CUDACHECK(cudaMemcpy(beta1s[i], &beta1, sizeof(T), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMalloc(beta2s + i, sizeof(T)));
    T beta2 = 0.5;
    CUDACHECK(cudaMemcpy(beta2s[i], &beta2, sizeof(T), cudaMemcpyHostToDevice));
    CUBLASCHECK(cublasSetPointerMode(handle[i], CUBLAS_POINTER_MODE_DEVICE));
    CUBLASCHECK(cublasSetStream(handle[i], s[i]));
    CUDACHECK(cudaMalloc(stepsizes + i, sizeof(T)));
    CUDACHECK(cudaMemcpy(stepsizes[i], &stepsize, sizeof(T), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMalloc(epsilons + i, sizeof(T)));
    T epsilon = 1e-6;
    CUDACHECK(cudaMemcpy(epsilons[i], &epsilon, sizeof(T), cudaMemcpyHostToDevice));
  }

  // //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  cudaEvent_t start[nDev], stop[nDev];
  float elapsedTime = 0.0;
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaEventCreate(&start[i]));
    CUDACHECK(cudaEventCreate(&stop[i]));
    CUDACHECK(cudaEventRecord(start[i],s[i]));
  }

  for (int iter = 0; iter < epochs; iter++) {
    if (check_results) {
      for (int dev = 0; dev < nDev; dev++) {
        CUDACHECK(cudaSetDevice(dev));
        CUDACHECK(cudaMemcpy(old_weights[dev], weights[dev], num_weight_elems*sizeof(T), 
                            cudaMemcpyDeviceToDevice));
        // CUDACHECK(cudaMemcpy(old_moments[dev], moments[dev], num_weight_elems*sizeof(T), 
        //                     cudaMemcpyDeviceToDevice));
        // CUDACHECK(cudaMemcpy(old_second_moments[dev], second_moments[dev], num_weight_elems*sizeof(T), 
        //                     cudaMemcpyDeviceToDevice));
      }
    }
    printf("iter %d\n", iter);

    switch (algo)
    {
      case 2:
        fused_allreduce_adam(nDev, devs, iter, comms, num_weight_elems,
                             minibatch_gradients, weights, moments, second_moments,
                             s, stepsizes, beta1s, beta2s, handle,
                             datatype);
        break;
      
        default:
          printf("Invalid algo %d", algo);
          break;
    }
    
    if (check_results)
      assert(check_epoch_results_adam(algo, iter, nDev, devs, num_weight_elems,
                        minibatch_gradients, allreduced_gradient, old_weights,
                        weights, h_moments, moments, h_second_moments, second_moments,
                        beta1s, beta2s, stepsizes, epsilons));
    
    //Swap gradients
    //swap(allreduced_gradient, minibatch_gradients);
  }
  
  for (int i = 0; i < nDev; i++) {
    float t;
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaEventRecord(stop[i],s[i]));
    CUDACHECK(cudaEventSynchronize(stop[i]));
    CUDACHECK(cudaEventElapsedTime(&t, start[i], stop[i]));
    elapsedTime = max(elapsedTime, t);
  }

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(minibatch_gradients[i]));
    CUDACHECK(cudaFree(allreduced_gradient[i]));
    CUDACHECK(cudaFree(weights[i]));
    CUBLASCHECK(cublasDestroy(handle[i]));
    CUDACHECK(cudaEventDestroy(stop[i]));
    CUDACHECK(cudaEventDestroy(start[i]));
    CUDACHECK(cudaStreamDestroy(s[i]));
  }

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);
  
  return elapsedTime;
}

template<class T>
void fused_allreduce_weight_update_scattered_weights(const int nDev, const int devs[],
                                              const ncclComm_t comms[],
                                              T** minibatch_gradients, 
                                              T*** weights,
                                              cudaStream_t* s,
                                              size_t nBuff, size_t **buffSizes,
                                              size_t sumBuffSizes,
                                              cublasHandle_t* handle,
                                              ncclDataType_t datatype)
{
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduceScatteredWeights((const void*)minibatch_gradients[i], 
                                     (void**)weights[i], nBuff, buffSizes[i],
                                     sumBuffSizes, nullptr, datatype, 
                                     ncclSum, comms[i], s[i]));
  }

  NCCLCHECK(ncclGroupEnd());
  
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }
}

template<class T>
void memset_value(T*f, T v, size_t nelems) 
{
  T* h_buff = new T[nelems];

  for (uint64_t i = 0; i < nelems; i++) {
    h_buff[i] = v;
  }

  CUDACHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  delete[] h_buff;
}

template<class T>
bool check_epoch_results_scattered_grad_weights(int algo, int epoch, const int nDev, const int devs[],
                                  T*** d_minibatch_gradients, 
                                  T*** h_weights, T*** d_new_weights, 
                                  size_t nbuffs, const size_t *h_buffSizes,
                                  T alpha)
                        
{
  printf("%s:%d checking\n", __FILE__, __LINE__);
  bool passed = true;
  T ***h_minibatch_gradients = (T***)malloc(nDev * sizeof(T**));
  // const size_t grad_array_size = num_weight_elems*sizeof(T);

  //Check AllReduced
  for (int dev = 0; dev < nDev; dev++) {
    CUDACHECK(cudaSetDevice(dev));
    h_minibatch_gradients[dev] = (T**)malloc(nbuffs*sizeof(T**));
    for (size_t buff = 0; buff < nbuffs; buff++) {
      h_minibatch_gradients[dev][buff] = (T*)malloc(h_buffSizes[buff]*sizeof(T));
      CUDACHECK(cudaMemcpy(h_minibatch_gradients[dev][buff], d_minibatch_gradients[dev][buff], 
                           h_buffSizes[buff]*sizeof(T), cudaMemcpyDeviceToHost));
      }
  }

  T **h_reduced_grad = (T**)malloc(sizeof(T*)*nbuffs);

  // CUDACHECK(cudaSetDevice(0));
  // CUDACHECK(cudaMemcpy(h_reduced_grad, d_allreduced_gradient[0], 
  //                      grad_array_size, cudaMemcpyDeviceToHost));

  for (size_t buff = 0; buff < nbuffs; buff++) {
    h_reduced_grad[buff] = (T*)malloc(h_buffSizes[buff]*sizeof(T));
    for (uint64_t i = 0; i < h_buffSizes[buff]; i++) {
      T sum = 0.0;

      for (int dev = 0; dev < nDev; dev++) {
        sum += h_minibatch_gradients[dev][buff][i];
        if (buff == 3 and i == 0) {
          printf("dev %d h_minibatch_gradients[dev][buff][i] %f\n", dev, h_minibatch_gradients[dev][buff][i]);
        }
      }

      h_reduced_grad[buff][i] = sum;

      if (buff == 1 and i == 0) {
        printf("h_reduced_grad[buff][i] %f\n", h_reduced_grad[buff][i]);
      }
    }
  }

  //Check Weight Update
  T ***h_new_weights = (T***)malloc(nDev * sizeof(T**));

  //Check AllReduced
  for (int dev = 0; dev < nDev; dev++) {
    CUDACHECK(cudaSetDevice(dev));
    h_new_weights[dev] = (T**)malloc(nbuffs*sizeof(T**));
    for (size_t buff = 0; buff < nbuffs; buff++) {
      h_new_weights[dev][buff] = (T*)malloc(h_buffSizes[buff]*sizeof(T));

      CUDACHECK(cudaMemcpy(h_new_weights[dev][buff], d_new_weights[dev][buff], 
                           h_buffSizes[buff]*sizeof(T), cudaMemcpyDeviceToHost));
    }
  }

  for (size_t buff = 0; buff < nbuffs; buff++) {
    for (uint64_t i = 0; i < h_buffSizes[buff]; i++) {
      for (int dev = 0; dev < nDev; dev++) {
        T new_weight = h_weights[dev][buff][i] + alpha *  h_reduced_grad[buff][i];
        if (!eq_float(new_weight, h_new_weights[dev][buff][i]) > 1e-4) {
          printf("Epoch %d Mismatch in h_new_weights for device %d in buff %ld at [%ld]: ref '%f' computed '%f'\n", epoch, dev, buff, i, new_weight, h_new_weights[dev][buff][i]);
          printf("h_weights[%ld] = %f , h_reduced_gradients[%ld] = %f\n", i, h_weights[dev][buff][i], i, h_reduced_grad[buff][i]);
          // for (int dev2 = 0; dev2 < nDev; dev2++) {
          //   printf("%d: %f, ", dev2, h_minibatch_gradients[dev2][i]);
          // }
          // printf("\n");
          passed = false;
        }
      }
      if (!passed)
        break;
    }
  }

  //Correct these to free
  //Remove frees, lets hope the system has significant RAM.

  // for (int i = 0; i < nDev; i++) {
  //   free(h_minibatch_gradients[i]);
  //   free(h_weights[i]);
  //   free(h_new_weights[i]);
  // }

  // free(h_new_weights);
  // free(h_reduced_grad);
  // free(h_weights);
  // free(h_minibatch_gradients);

  return passed;
}

template<class T>
float run_scattered_weights(int algo, bool check_results, uint64_t nbuffs, const size_t h_buffSizes[], T gradient_factor, 
          ncclDataType_t datatype, double& weight_update_time, double& reduce_time)
{
  //This code implements the backward pass of a large layer, using model parallelism
  //The whole dataset is divided into several minibatches, with one minibatch assigned to one gpu.
  //The gradients are received from each gpu, then reduced, and weights are updated as:
  //w = w - alpha * g
  
  //managing 4 devices
  const int nDev = 16;
  int devs[nDev] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  ncclComm_t comms[nDev];
  const int epochs = 10;

  //allocating and initializing device buffers
  T** minibatch_gradients = (T**)malloc(nDev * sizeof(T*));
  T*** allreduced_gradient  = (T***)malloc(nDev * sizeof(T*));
  T*** weights = (T***)malloc(nDev * sizeof(T*));
  T*** old_weights = (T***)malloc(nDev * sizeof(T*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  // T** alphas = (T**)malloc(nDev * sizeof(T*));
  // T** betas = (T**)malloc(nDev * sizeof(T*));
  cublasHandle_t* handle = (cublasHandle_t*)malloc(nDev * sizeof(cublasHandle_t));
  T*** h_weights = (T***)malloc(nDev*sizeof(T));
  T** h_minibatch_gradients = (T**)malloc(nDev*sizeof(T*));
  T*** h_old_weights = (T***)malloc(nDev*sizeof(T));
  size_t** buffSizes = (size_t**)malloc(nDev * sizeof(size_t*));
  size_t totalSize = 0;
  for (size_t buff = 0; buff < nbuffs; buff++) {
    totalSize += h_buffSizes[buff];
  }

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    //buffSizes[i] = (size_t**)malloc(nBuffs * sizeof(size_t*));
    CUDACHECK(cudaMalloc(&buffSizes[i], nbuffs * sizeof(size_t)));
    CUDACHECK(cudaMemcpy(buffSizes[i], h_buffSizes, nbuffs * sizeof(size_t), cudaMemcpyHostToDevice));
    T** weights_tmp = (T**)malloc(nbuffs * sizeof(T*));
    h_weights[i] = weights_tmp;
    // T** minibatch_gradients_tmp = (T**)malloc(nbuffs * sizeof(T*));
    // h_minibatch_gradients[i] = minibatch_gradients_tmp;

    T** old_weights_tmp = (T**)malloc(nbuffs * sizeof(T*));
    h_old_weights[i] = old_weights_tmp;

    for (size_t buff = 0; buff < nbuffs; buff++) {
      
      // CUDACHECK(cudaMalloc(&minibatch_gradients_tmp[buff], h_buffSizes[buff] * sizeof(T)));
      
      CUDACHECK(cudaMalloc(&weights_tmp[buff], h_buffSizes[buff] * sizeof(T)));
      h_old_weights[i][buff] = (T*)malloc(h_buffSizes[buff] * sizeof(T));
      //CUDACHECK(cudaMalloc(&old_weights_tmp[buff], h_buffSizes[buff] * sizeof(T)));      
      //cudaMemRandInt(minibatch_gradients_tmp[buff], h_buffSizes[buff]);
      // memset_value(minibatch_gradients_tmp[buff], (float)(1), h_buffSizes[buff]);
      // if (i == 0) {
      //   h_weights[buff] = (T*)malloc(size*sizeof(T));
      //   cudaMemRandInt(weights[buff][i], num_weight_elems);
      //   CUDACHECK(cudaMemcpy(h_weights[buff], weights[buff][i], num_weight_elems * sizeof(T), cudaMemcpyDeviceToHost));
      // } else {
      //   CUDACHECK(cudaMemcpy(weights[buff][i], h_weights[buff], num_weight_elems * sizeof(T), cudaMemcpyHostToDevice));
      // }
      
      //CUDACHECK(cudaMemset(weights_tmp[buff], 0, h_buffSizes[buff] * sizeof(T)));
      
      //Parameters for cublas
      // CUBLASCHECK(cublasCreate(handle + i));
      // T alpha = 1.0;
      // CUDACHECK(cudaMalloc(alphas + i, sizeof(T)));
      // CUDACHECK(cudaMemcpy(alphas[i], &alpha, sizeof(T), cudaMemcpyHostToDevice));
      // CUDACHECK(cudaMalloc(betas + i, sizeof(T)));
      // T beta = -gradient_factor;
      // CUDACHECK(cudaMemcpy(betas[i], &beta, sizeof(T), cudaMemcpyHostToDevice));
      // CUBLASCHECK(cublasSetPointerMode(handle[i], CUBLAS_POINTER_MODE_DEVICE));
      // CUBLASCHECK(cublasSetStream(handle[i], s[i]));
    }

    CUDACHECK(cudaMalloc(&minibatch_gradients[i], totalSize * sizeof(T)));
    //memset_value(minibatch_gradients[i], (float)(1), totalSize);
    cudaMemRandInt( minibatch_gradients[i], totalSize);
    printf("dev %d ", i);
    for(uint64_t buff = 0; buff < nbuffs; buff++) {
        printf("weights_tmp[%ld] %p ", buff, weights_tmp[buff]);
    }
    printf("\n");

    CUDACHECK(cudaMalloc(&weights[i], nbuffs * sizeof(T**)));
    CUDACHECK(cudaMemcpy(weights[i], weights_tmp, nbuffs * sizeof(T**), cudaMemcpyHostToDevice));

    // CUDACHECK(cudaMalloc(&minibatch_gradients[i], nbuffs * sizeof(T**)));
    // CUDACHECK(cudaMemcpy(minibatch_gradients[i], minibatch_gradients_tmp, nbuffs * sizeof(T**), cudaMemcpyHostToDevice));

    CUDACHECK(cudaStreamCreate(s+i));
  }

  // //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  cudaEvent_t start[nDev], stop[nDev];
  float elapsedTime = 0.0;
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaEventCreate(&start[i]));
    CUDACHECK(cudaEventCreate(&stop[i]));
    CUDACHECK(cudaEventRecord(start[i],s[i]));
  }

  for (int iter = 0; iter < epochs; iter++) {
    if (check_results) {
      for (int dev = 0; dev < nDev; dev++) {
        CUDACHECK(cudaSetDevice(dev));
        cudaMemRandInt( minibatch_gradients[dev], totalSize);
        for (size_t buff = 0; buff < nbuffs; buff++) {
          CUDACHECK(cudaMemcpy(h_old_weights[dev][buff], h_weights[dev][buff], h_buffSizes[buff]*sizeof(T), 
                               cudaMemcpyDeviceToHost));
        }
      }
    }

    switch (algo)
    {
      // case 0:
      //   traditional_weight_update(nDev, devs, comms, num_weight_elems,
      //                             minibatch_gradients, allreduced_gradient, weights,
      //                             s, alphas, betas, handle,
      //                             datatype, weight_update_time, reduce_time);
      //   break;
      // case 1:
      //   allreduce_async_weight_update(nDev, devs, comms, num_weight_elems,
      //                                 minibatch_gradients, allreduced_gradient, weights,
      //                                 s, alphas, betas, handle,
      //                                 datatype);
      //     break;
      // (const int nDev, const int devs[],
      //   const ncclComm_t comms[],
      //   T*** minibatch_gradients, 
      //   T*** weights,
      //   cudaStream_t* s,
      //   T*** gradient_factors, size_t* nBuff, size_t ***buffSizes,
      //   cublasHandle_t* handle,
      //   ncclDataType_t datatype)

      case 2:
        fused_allreduce_weight_update_scattered_weights(nDev, devs, comms,
                                      minibatch_gradients, weights,
                                      s, nbuffs, buffSizes, totalSize, handle,
                                      datatype);
        break;
      
        default:
          printf("Invalid algo %d", algo);
          break;
    }
    
    if (check_results)
      assert(check_epoch_results_scattered(algo, iter, nDev, devs,
                                          minibatch_gradients, h_old_weights,
                                          h_weights, nbuffs, h_buffSizes));
    
    //Swap gradients
    // swap(allreduced_gradient, minibatch_gradients);
  }
  

  for (int i = 0; i < nDev; i++) {
    float t;
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaEventRecord(stop[i],s[i]));
    CUDACHECK(cudaEventSynchronize(stop[i]));
    CUDACHECK(cudaEventElapsedTime(&t, start[i], stop[i]));
    elapsedTime = max(elapsedTime, t);
  }

  #if 0
  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(minibatch_gradients[i]));
    CUDACHECK(cudaFree(allreduced_gradient[i]));
    CUDACHECK(cudaFree(weights[i]));
    CUDACHECK(cudaFree(alphas[i]));
    CUDACHECK(cudaFree(betas[i]));
    CUBLASCHECK(cublasDestroy(handle[i]));
    CUDACHECK(cudaEventDestroy(stop[i]));
    CUDACHECK(cudaEventDestroy(start[i]));
    CUDACHECK(cudaStreamDestroy(s[i]));
  }
  #endif

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);
  
  return elapsedTime;
}


template<class T>
void fused_allreduce_weight_update_scattered_grad_weights(const int nDev, const int devs[],
                                              const ncclComm_t comms[],
                                              T*** minibatch_gradients, 
                                              T*** weights,
                                              T** alphas,
                                              cudaStream_t* s,
                                              size_t nBuff, size_t **buffSizes,
                                              size_t sumBuffSizes,
                                              cublasHandle_t* handle,
                                              ncclDataType_t datatype)
{

  std::cout <<"sumBuffSizes: "<<sumBuffSizes<<std::endl;
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    NCCLCHECK(ncclAllReduceScatteredGradWeights((const void**)minibatch_gradients[i], 
                                     (void**)weights[i], nBuff, buffSizes[i],
                                     sumBuffSizes, alphas[i], datatype, 
                                     ncclSum, comms[i], s[i]));
  }

  NCCLCHECK(ncclGroupEnd());
  
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }
}

template<class T>
float run_scattered_grads_weights(int algo, bool check_results, uint64_t nbuffs, const size_t h_buffSizes[], T gradient_factor, 
          ncclDataType_t datatype, double& weight_update_time, double& reduce_time)
{
  //This code implements the backward pass of a large layer, using model parallelism
  //The whole dataset is divided into several minibatches, with one minibatch assigned to one gpu.
  //The gradients are received from each gpu, then reduced, and weights are updated as:
  //w = w - alpha * g
  
  //managing 4 devices
  //const int nDev = 16;
  //int devs[nDev] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  const int nDev = 4;
  int devs[nDev] = { 0, 1, 2, 3 };
  ncclComm_t comms[nDev];
  const int epochs = 10;

  //allocating and initializing device buffers
  T*** minibatch_gradients = (T***)malloc(nDev * sizeof(T**));
  T*** allreduced_gradient  = (T***)malloc(nDev * sizeof(T*));
  T*** weights = (T***)malloc(nDev * sizeof(T*));
  T*** old_weights = (T***)malloc(nDev * sizeof(T*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  T** alphas = (T**)malloc(nDev * sizeof(T*));
  cublasHandle_t* handle = (cublasHandle_t*)malloc(nDev * sizeof(cublasHandle_t));
  T*** h_weights = (T***)malloc(nDev*sizeof(T));
  T*** h_minibatch_gradients = (T***)malloc(nDev*sizeof(T**));
  T*** h_old_weights = (T***)malloc(nDev*sizeof(T));
  size_t** buffSizes = (size_t**)malloc(nDev * sizeof(size_t*));
  size_t totalSize = 0;
  for (size_t buff = 0; buff < nbuffs; buff++) {
    totalSize += h_buffSizes[buff];
  }

  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    //buffSizes[i] = (size_t**)malloc(nBuffs * sizeof(size_t*));
    CUDACHECK(cudaMalloc(&buffSizes[i], nbuffs * sizeof(size_t)));
    CUDACHECK(cudaMemcpy(buffSizes[i], h_buffSizes, nbuffs * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMalloc(&alphas[i], sizeof(T)));
    T** weights_tmp = (T**)malloc(nbuffs * sizeof(T*));
    h_weights[i] = weights_tmp;
    T** minibatch_gradients_tmp = (T**)malloc(nbuffs * sizeof(T*));
    h_minibatch_gradients[i] = minibatch_gradients_tmp;

    T** old_weights_tmp = (T**)malloc(nbuffs * sizeof(T*));
    h_old_weights[i] = old_weights_tmp;

    for (size_t buff = 0; buff < nbuffs; buff++) {
      
      CUDACHECK(cudaMalloc(&minibatch_gradients_tmp[buff], h_buffSizes[buff] * sizeof(T)));
      
      CUDACHECK(cudaMalloc(&weights_tmp[buff], h_buffSizes[buff] * sizeof(T)));
      h_old_weights[i][buff] = (T*)malloc(h_buffSizes[buff] * sizeof(T));
      //CUDACHECK(cudaMalloc(&old_weights_tmp[buff], h_buffSizes[buff] * sizeof(T)));      
      cudaMemRandInt( minibatch_gradients_tmp[buff], h_buffSizes[buff]);
      //memset_value(minibatch_gradients_tmp[buff], (float)(1), h_buffSizes[buff]);
      // if (i == 0) {
      //   h_weights[buff] = (T*)malloc(size*sizeof(T));
      //   cudaMemRandInt(weights[buff][i], num_weight_elems);
      //   CUDACHECK(cudaMemcpy(h_weights[buff], weights[buff][i], num_weight_elems * sizeof(T), cudaMemcpyDeviceToHost));
      // } else {
      //   CUDACHECK(cudaMemcpy(weights[buff][i], h_weights[buff], num_weight_elems * sizeof(T), cudaMemcpyHostToDevice));
      // }
      
      //CUDACHECK(cudaMemset(weights_tmp[buff], 0, h_buffSizes[buff] * sizeof(T)));
      memset_value(weights_tmp[buff], 1.0f, h_buffSizes[buff]);
      //Parameters for cublas
      // CUBLASCHECK(cublasCreate(handle + i));
      // T alpha = 1.0;
      // CUDACHECK(cudaMalloc(alphas + i, sizeof(T)));
      // CUDACHECK(cudaMemcpy(alphas[i], &alpha, sizeof(T), cudaMemcpyHostToDevice));
      // CUDACHECK(cudaMalloc(betas + i, sizeof(T)));
    }

    CUDACHECK(cudaMemcpy(alphas[i], &gradient_factor, sizeof(T), cudaMemcpyHostToDevice));
    
    CUDACHECK(cudaMalloc(&minibatch_gradients[i], nbuffs * sizeof(T**)));
    //memset_value(minibatch_gradients[i], (float)(1), totalSize);

    printf("dev %d ", i);
    for(uint64_t buff = 0; buff < nbuffs; buff++) {
        printf("weights[%ld] %p ", buff, weights_tmp[buff]);
    }
    printf("\n");
    printf("dev %d ", i);
    for(uint64_t buff = 0; buff < nbuffs; buff++) {
        printf("minibatch_gradients[%ld] %p ", buff, minibatch_gradients_tmp[buff]);
    }
    printf("\n");

    CUDACHECK(cudaMalloc(&weights[i], nbuffs * sizeof(T**)));
    CUDACHECK(cudaMemcpy(weights[i], weights_tmp, nbuffs * sizeof(T**), cudaMemcpyHostToDevice));

    CUDACHECK(cudaMalloc(&minibatch_gradients[i], nbuffs * sizeof(T**)));
    CUDACHECK(cudaMemcpy(minibatch_gradients[i], minibatch_gradients_tmp, nbuffs * sizeof(T**), cudaMemcpyHostToDevice));

    CUDACHECK(cudaStreamCreate(s+i));
  }

  // //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  cudaEvent_t start[nDev], stop[nDev];
  float elapsedTime = 0.0;
  for (int i = 0; i < nDev; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaEventCreate(&start[i]));
    CUDACHECK(cudaEventCreate(&stop[i]));
    CUDACHECK(cudaEventRecord(start[i],s[i]));
  }

  for (int iter = 0; iter < epochs; iter++) {
    if (check_results) {
      for (int dev = 0; dev < nDev; dev++) {
        CUDACHECK(cudaSetDevice(dev));
        // cudaMemRandInt( minibatch_gradients[dev], totalSize);
        for (size_t buff = 0; buff < nbuffs; buff++) {
          CUDACHECK(cudaMemcpy(h_old_weights[dev][buff], h_weights[dev][buff], h_buffSizes[buff]*sizeof(T), 
                               cudaMemcpyDeviceToHost));
        }
      }
    }

    switch (algo)
    {
      // case 0:
      //   traditional_weight_update(nDev, devs, comms, num_weight_elems,
      //                             minibatch_gradients, allreduced_gradient, weights,
      //                             s, alphas, betas, handle,
      //                             datatype, weight_update_time, reduce_time);
      //   break;
      // case 1:
      //   allreduce_async_weight_update(nDev, devs, comms, num_weight_elems,
      //                                 minibatch_gradients, allreduced_gradient, weights,
      //                                 s, alphas, betas, handle,
      //                                 datatype);
      //     break;
      // (const int nDev, const int devs[],
      //   const ncclComm_t comms[],
      //   T*** minibatch_gradients, 
      //   T*** weights,
      //   cudaStream_t* s,
      //   T*** gradient_factors, size_t* nBuff, size_t ***buffSizes,
      //   cublasHandle_t* handle,
      //   ncclDataType_t datatype)

      case 2:
        fused_allreduce_weight_update_scattered_grad_weights(nDev, devs, comms,
                                      minibatch_gradients, weights, alphas,
                                      s, nbuffs, buffSizes, totalSize, handle,
                                      datatype);
        break;
      
        default:
          printf("Invalid algo %d", algo);
          break;
    }
    
    if (check_results)
      assert(check_epoch_results_scattered_grad_weights(algo, iter, nDev, devs,
                                          h_minibatch_gradients, h_old_weights,
                                          h_weights, nbuffs, h_buffSizes, gradient_factor));
    
    //Swap gradients
    // swap(allreduced_gradient, minibatch_gradients);
  }
  

  for (int i = 0; i < nDev; i++) {
    float t;
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaEventRecord(stop[i],s[i]));
    CUDACHECK(cudaEventSynchronize(stop[i]));
    CUDACHECK(cudaEventElapsedTime(&t, start[i], stop[i]));
    elapsedTime = max(elapsedTime, t);
  }

  #if 0
  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(minibatch_gradients[i]));
    CUDACHECK(cudaFree(allreduced_gradient[i]));
    CUDACHECK(cudaFree(weights[i]));
    CUDACHECK(cudaFree(alphas[i]));
    CUDACHECK(cudaFree(betas[i]));
    CUBLASCHECK(cublasDestroy(handle[i]));
    CUDACHECK(cudaEventDestroy(stop[i]));
    CUDACHECK(cudaEventDestroy(start[i]));
    CUDACHECK(cudaStreamDestroy(s[i]));
  }
  #endif

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);
  
  return elapsedTime;
}

int main(int argc, char* argv[])
{
  //This code implements the backward pass of a large layer, using model parallelism
  //The whole dataset is divided into several minibatches, with one minibatch assigned to one gpu.
  //The gradients are received from each gpu, then reduced, and weights are updated as:
  //w = w - alpha * g
  //Before running this program do "export NCCL_PROTO=LL"

  if (argc < 9) {
    printf ("Provide four arguments: algo check_results size type stepsize adam/sgd scattered/single perform_eval\n");
    return 0;
  }
  
  const int algo = atoi(argv[1]);  
  const int check_results = atoi(argv[2]);
  const int size = atoi(argv[3]);
  const char* type = argv[4];
  const double gradient_factor = strtod(argv[5], NULL);
  const char* opt_type = argv[6];
  const char* mem_type = argv[7];
  const int eval = atoi(argv[8]);
  printf("Using algo %d, check_results %d num elements %d opt_type %s mem_type %s doingeval? %d\n", algo, check_results, size, opt_type, mem_type, eval);

  if (eval == 0) {
    float elapsedTime;
    double weight_update_time = 0.0, reduce_time = 0.0;
    if (strcmp(type, "float") == 0) {
      if (strcmp(opt_type, "sgd") == 0) {
        if(strcmp(mem_type, "scattered") == 0) {
          //const size_t buffSizes[nbuffs] = {2048, 8192, 16384, 256};
          const size_t nbuffs = 1;
          size_t buffSizes[nbuffs];
          for (size_t i = 0; i < nbuffs; i++) 
            buffSizes[i] = (size_t)size/nbuffs;
          // const size_t nbuffs = 1;
          // const size_t buffSizes[nbuffs] = {(size_t) size};
/*          const size_t nbuffs = 397;
          const size_t buffSizes[nbuffs] = {31260672,
		          524288,
			  2048,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  1048576,
			  1048576,
			  4194304,
			  4194304,
			  1048576,
			  1048576,
			  2048,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  1024,
			  4096,
			  1024,
			  1024,
			  1024,
			  1024,
			  30528,
			  1024,
			  1024,
			  1024
	  };*/
          //const size_t buffSizes[nbuffs] = {128, 256, 512, 128};
          //elapsedTime = run_scattered_weights<float>(algo, (bool)check_results, nbuffs, buffSizes, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);
          elapsedTime = run_scattered_grads_weights<float>(algo, (bool)check_results, nbuffs, buffSizes, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);
          
          // const size_t buffSizes2[nbuffs] = {128, 256, 512, 128};
          // //elapsedTime = run_scattered_weights<float>(algo, (bool)check_results, nbuffs, buffSizes, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);
          // elapsedTime = run_scattered_grads_weights<float>(algo, (bool)check_results, nbuffs, buffSizes2, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);

        }
        else
          elapsedTime = run<float>(algo, (bool)check_results, size, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);
      }
      else if (strcmp(opt_type, "adam") == 0)
        elapsedTime = run_adam<float>(algo, (bool)check_results, size, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);
      else {
        printf("Unknown optimizer\n");
        assert(false);
      }
    } else if (strcmp(type, "double") == 0) {
      elapsedTime = run<double>(algo, (bool)check_results, size, (double)gradient_factor, ncclDouble, weight_update_time, reduce_time);
    } else {
      printf("Invalid type '%s'\n", type);
    }

    printf ("Elapsed Time: %f\n", elapsedTime);
  } else if(true) {
    if(check_results != 0) {
      printf("No need to check results during evaluation\n");
      return 0;
    }
    
    printf("%-10s %-10s %-10s %-10s\n","Size","Baseline(ms)","Fused(ms)","Speedup");
    for (uint64_t size = 128; size <= 1024*1024*1024; size = size * 4) {
      double weight_update_time = 0.0, reduce_time = 0.0;
      float elapsedTime1 = 0;
      printf("%-15.2ld", size);
      if (strcmp(type, "float") == 0) {
        elapsedTime1 = run<float>(0, false, size, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);
      } else if (strcmp(type, "double") == 0) {
        elapsedTime1 = run<double>(0, false, size, (double)gradient_factor, ncclDouble, weight_update_time, reduce_time);
      } else {
        printf("Invalid type '%s'\n", type);
        return 0;
      }
      printf("%-15.2f", elapsedTime1);
      float elapsedTime2 = 0.0;
      if (strcmp(type, "float") == 0) {
        elapsedTime2 = run<float>(2, false, size, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);
      } else if (strcmp(type, "double") == 0) {
        elapsedTime2 = run<double>(2, false, size, (double)gradient_factor, ncclDouble, weight_update_time, reduce_time);
      } else {
        printf("Invalid type '%s'\n", type);
        return 0;
      }

      printf("%-15.2f", elapsedTime2);
      printf("%-15.2f\n", elapsedTime1/elapsedTime2);
    }
  } else {
    printf("%-10s %-10s %-10s\n","Size","FracWeightUpdateTime","FracAllReducedTime");
    for (uint64_t size = 128; size <= 1024*1024*1024; size = size * 4) {
      //uint64_t size = 512*1024*1024;
      double weight_update_time = 0.0, reduce_time = 0.0;
      float elapsedTime1 = 0;
      printf("%-15.2ld", size);
      if (strcmp(type, "float") == 0) {
        elapsedTime1 = run<float>(0, false, size, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);
      } else if (strcmp(type, "double") == 0) {
        elapsedTime1 = run<double>(0, false, size, (double)gradient_factor, ncclDouble, weight_update_time, reduce_time);
      } else {
        printf("Invalid type '%s'\n", type);
        return 0;
      }

      printf("%-15.2f", weight_update_time/elapsedTime1);
      printf("%-15.2f\n", reduce_time/elapsedTime1);
    }
  }

  return 0;
}
