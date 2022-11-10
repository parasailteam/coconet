#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cstdint>
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
bool check_epoch_results_adam(int algo, int buffIndex, int epoch, const int nDev, const int devs[],
                        const uint64_t num_weight_elems,
                        T** h_minibatch_gradients, 
                        // T** d_allreduced_gradient, 	
                        T** h_weights, T** d_new_weights, 
                        T* cpu_moment,
                        T* cpu_second_moment,
                        T beta1, T beta2, T stepsize, T epsilon)	
{
  bool passed = true;
//   T **h_minibatch_gradients = (T**)malloc(nDev * sizeof(T*));
  const size_t grad_array_size = num_weight_elems*sizeof(T);

//   //Check AllReduced
//   for (int dev = 0; dev < nDev; dev++) {
//     CUDACHECK(cudaSetDevice(dev));
//     h_minibatch_gradients[dev] = (T*)malloc(num_weight_elems*sizeof(T));
//     CUDACHECK(cudaMemcpy(h_minibatch_gradients[dev], d_minibatch_gradients[dev], 
//                          grad_array_size, cudaMemcpyDeviceToHost));
//   }

  T *h_reduced_grad = (T*)malloc(grad_array_size);


  for (uint64_t i = 0; i < num_weight_elems; i++) {
    T sum = 0.0;

    for (int dev = 0; dev < nDev; dev++) {
      sum += h_minibatch_gradients[dev][i];
    }

	h_reduced_grad[i] = sum;
  }

  //Check Weight Update

  T **h_all_reduced_grads = (T**)malloc(nDev * sizeof(T*));
  T **h_old_moments = (T**)malloc(nDev * sizeof(T*));
  T **h_old_second_moments = (T**)malloc(nDev * sizeof(T*));

  T **h_new_weights = (T**)malloc(sizeof(T*)*nDev);

  for (int i = 0; i < nDev; i++) {
    T *ptr = (T*)malloc(grad_array_size);
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemcpy(ptr, d_new_weights[i], 
                         grad_array_size, cudaMemcpyDeviceToHost));
    h_new_weights[i] = ptr;
  }

  CUDACHECK(cudaSetDevice(0));
  

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

      T old_v = cpu_second_moment[i];
      v = beta2 * old_v + (1-beta2) * h_reduced_grad[i]*h_reduced_grad[i];
      
      T m_ = m/(1 - pow(beta1, epoch + 1));
      T v_ = v/(1 - pow(beta2, epoch + 1));
      T x = stepsize * m_ / (sqrt(v_) + epsilon);
      T new_weight = h_weights[dev][i] + x;

      if (!eq_float(new_weight, h_new_weights[dev][i])) {
        //Lets take a look at the last device only.
		printf("BuffIndex %d Epoch %d Mismatch in h_new_weights for device %d at [%ld]: ref '%f' computed '%f'\n", buffIndex, epoch, dev, i, new_weight, h_new_weights[dev][i]);
		printf("x: correct %f != incorrect %f, m_ %f, v_ %f, beta1 %f, beta2 %f, old_m %f, old_v %f\n", x, h_new_weights[dev][i], m_, v_, beta1, beta2, old_m, old_v);
        // printf("h_weights[%ld] = %f , h_reduced_gradients[%ld] = %f, new_m = %f, new_v = %f, m_ = %f, v_ = %f, old_m = %f, old_v = %f d_new_m = %f\n", i, h_weights[dev][i], i, h_reduced_grad[i], m, v, m_, v_, old_m, old_v, h_moments[dev][i]);
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
float run_scattered_adam(uint64_t nbuffs, const size_t* h_buffSizes, ncclDataType_t datatype, bool check_results = true)
{
	const int nDev = 4;
	int devs[nDev] = { 0, 1, 2, 3 };
	ncclComm_t comms[nDev];
	const int epochs = 10;

	T*** gradients  = (T***)malloc(nDev * sizeof(T**));
	T*** weights = (T***)malloc(nDev * sizeof(T**));
	T*** firstMoment  = (T***)malloc(nDev * sizeof(T**));
	T*** secondMoment = (T***)malloc(nDev * sizeof(T**));
	T** alphas = (T**)malloc(nDev * sizeof(T*));
	T** beta1s = (T**)malloc(nDev * sizeof(T*));
	T** beta2s = (T**)malloc(nDev * sizeof(T*));

	const T beta1 = (T)0.5f;
	const T beta2 = (T)0.5f;
	const T alpha = (T)1.0f;

	cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
	//   cublasHandle_t* handle = (cublasHandle_t*)malloc(nDev * sizeof(cublasHandle_t));
	//   T** h_weights = (T**)malloc(nDev*sizeof(T));
	//   T*** h_gradients = (T***)malloc(nDev*sizeof(T*));
	size_t** buffSizes = (size_t**)malloc(nDev * sizeof(size_t*));
	size_t totalSize = 0;
	for (size_t buff = 0; buff < nbuffs; buff++) {
		totalSize += h_buffSizes[buff];
	}
	// for (size_t buff = 0; buff < nbuffs; buff++) {
	// 	h_weights = (T*)malloc(h_buffSizes[buff] * sizeof(T));
	// }

	T*** tmpGrads = (T***)malloc(nDev * sizeof(T**));
	T*** tmpWeights = (T***)malloc(nDev * sizeof(T**));
	T*** tmpFirstMoment = (T***)malloc(nDev * sizeof(T**));
	T*** tmpSecondMoment = (T***)malloc(nDev * sizeof(T**));

	T** cpuMomentum = (T**)malloc(nbuffs * sizeof(T*));
	T** cpuVelocity = (T**)malloc(nbuffs * sizeof(T*));
	T** cpuGrads = (T**)malloc(nbuffs * sizeof(T*));
	T** cpuWeight = (T**)malloc(nbuffs * sizeof(T*));
	for (int i = 0; i < nbuffs; i++) {
		cpuVelocity[i] = (T*)malloc(h_buffSizes[i]*sizeof(T));
		cpuMomentum[i] = (T*)malloc(h_buffSizes[i]*sizeof(T));
		cpuGrads[i] = (T*)malloc(h_buffSizes[i]*sizeof(T));
		cpuWeight[i] = (T*)malloc(h_buffSizes[i]*sizeof(T));
		for (int index = 0; index < h_buffSizes[i]; index++){
			cpuVelocity[i][index] = 0.f;
			cpuMomentum[i][index] = 0.f;
			cpuGrads[i][index] = 1.f;
			cpuWeight[i][index] = 0.f;
		}
	}

	for (int i = 0; i < nDev; ++i) {
		CUDACHECK(cudaSetDevice(devs[i]));

		CUDACHECK(cudaMalloc(&buffSizes[i], nbuffs * sizeof(size_t)));
		CUDACHECK(cudaMemcpy(buffSizes[i], h_buffSizes, nbuffs * sizeof(size_t), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMalloc(&weights[i], nbuffs * sizeof(T*)));
		CUDACHECK(cudaMalloc(&gradients[i], nbuffs * sizeof(T*)));
		CUDACHECK(cudaMalloc(&firstMoment[i], nbuffs * sizeof(T*)));
		CUDACHECK(cudaMalloc(&secondMoment[i], nbuffs * sizeof(T*)));

		tmpWeights[i] = (T**) malloc(nbuffs * sizeof(T*));
		tmpGrads[i] = (T**) malloc(nbuffs * sizeof(T*));
		tmpFirstMoment[i] = (T**) malloc(nbuffs * sizeof(T*));
		tmpSecondMoment[i] = (T**) malloc(nbuffs * sizeof(T*));

		CUDACHECK(cudaMalloc(&alphas[i], sizeof(T)));
		CUDACHECK(cudaMalloc(&beta1s[i], sizeof(T)));
		CUDACHECK(cudaMalloc(&beta2s[i], sizeof(T)));
		
		CUDACHECK(cudaMemcpy(alphas[i], &alpha, sizeof(T), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(beta1s[i], &beta1, sizeof(T), cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(beta2s[i], &beta2, sizeof(T), cudaMemcpyHostToDevice));

		for (size_t buff = 0; buff < nbuffs; buff++) {
			CUDACHECK(cudaMalloc(&tmpGrads[i][buff], h_buffSizes[buff] * sizeof(T)));
			CUDACHECK(cudaMalloc(&tmpWeights[i][buff], h_buffSizes[buff] * sizeof(T)));

			CUDACHECK(cudaMemcpy(tmpGrads[i][buff], cpuGrads[buff], h_buffSizes[buff] * sizeof(T), cudaMemcpyHostToDevice));
			CUDACHECK(cudaMemcpy(tmpWeights[i][buff], cpuWeight[buff], h_buffSizes[buff] * sizeof(T), cudaMemcpyHostToDevice));

			CUDACHECK(cudaMalloc(&tmpFirstMoment[i][buff], h_buffSizes[buff] * sizeof(T)));
			CUDACHECK(cudaMalloc(&tmpSecondMoment[i][buff], h_buffSizes[buff] * sizeof(T)));

			CUDACHECK(cudaMemcpy(tmpFirstMoment[i][buff], cpuMomentum[buff], h_buffSizes[buff] * sizeof(T), cudaMemcpyHostToDevice));
			CUDACHECK(cudaMemcpy(tmpSecondMoment[i][buff], cpuVelocity[buff], h_buffSizes[buff] * sizeof(T), cudaMemcpyHostToDevice));
		}

		CUDACHECK(cudaMemcpy(gradients[i], tmpGrads[i], sizeof(T*) * nbuffs, cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(weights[i], tmpWeights[i], sizeof(T*) * nbuffs, cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(firstMoment[i], tmpFirstMoment[i], sizeof(T*) * nbuffs, cudaMemcpyHostToDevice));
		CUDACHECK(cudaMemcpy(secondMoment[i], tmpSecondMoment[i], sizeof(T*) * nbuffs, cudaMemcpyHostToDevice));
		CUDACHECK(cudaStreamCreate(s+i));
	}
	// free(tmpGrads);
	// free(tmpWeights);
	// free(tmpFirstMoment);
	// free(tmpSecondMoment);


	// //initializing NCCL
	NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

	cudaEvent_t start[nDev], stop[nDev];
	float elapsedTime = 0.0;
	for (int i = 0; i < nDev; i++) {
		CUDACHECK(cudaSetDevice(devs[i]));
		CUDACHECK(cudaEventCreate(&start[i]));
		CUDACHECK(cudaEventCreate(&stop[i]));
		CUDACHECK(cudaEventRecord(start[i],s[i]));
	}  

	T*** tmpOldWeights = new T**[nbuffs];
	T*** tmpOldGrads = new T**[nbuffs];
	for (int buff = 0; buff < nbuffs; buff++) {
		tmpOldWeights[buff] = new T*[nDev];
		tmpOldGrads[buff] = new T*[nDev];
		for (int d = 0; d < nDev; d++) {
			tmpOldWeights[buff][d] = new T[h_buffSizes[buff]];
			tmpOldGrads[buff][d] = new T[h_buffSizes[buff]];
		}
	}
	printf("Execution starting...\n");
	for (int iter = 0; iter < epochs; iter++) {
		if (check_results) {
			for (int buff = 0; buff < nbuffs; buff++) {
				for (int d = 0; d < nDev; d++) {
					cudaMemcpy(tmpOldWeights[buff][d], tmpWeights[d][buff], sizeof(T)*h_buffSizes[buff], cudaMemcpyDeviceToHost);
					cudaMemcpy(tmpOldGrads[buff][d], tmpGrads[d][buff], sizeof(T)*h_buffSizes[buff], cudaMemcpyDeviceToHost);
				}
			}
		}

		NCCLCHECK(ncclGroupStart());
		for (int i = 0; i < nDev; ++i) {
//			NCCLCHECK(ncclAllReduce(tmpGrads[i][0], tmpWeights[i][0], totalSize, datatype, ncclSum, comms[i], s[i]));
			NCCLCHECK(ncclAllReduceScatteredAdam((const void**)gradients[i], (void**)weights[i], (void**)firstMoment[i], (void**)secondMoment[i], nbuffs, buffSizes[i],
						totalSize, (void*) alphas[i], (void*) beta1s[i], (void*)beta2s[i], iter, datatype, ncclSum, comms[i], s[i]));
		}

		NCCLCHECK(ncclGroupEnd());

		for (int i = 0; i < nDev; ++i) {
			CUDACHECK(cudaSetDevice(devs[i]));
			CUDACHECK(cudaStreamSynchronize(s[i]));
		}

		if (check_results) {
			T** h_gradients = new T*[nDev];
			T** h_new_weights = new T*[nDev];
			
			for (int buff = 0; buff < nbuffs; buff++) {
				for (int d = 0; d < nDev; d++) {
					// h_gradients[d] = tmpGrads[d][buff];
					h_new_weights[d] = tmpWeights[d][buff];
				}

				check_epoch_results_adam(2, buff, iter, nDev, devs, h_buffSizes[buff], tmpOldGrads[buff], tmpOldWeights[buff], h_new_weights, cpuMomentum[buff], cpuVelocity[buff], beta1, beta2, alpha, (T)1e-6);
			}
		}
		
	}
						
	for (int i = 0; i < nDev; i++) {
		float t;
		CUDACHECK(cudaSetDevice(devs[i]));
		CUDACHECK(cudaEventRecord(stop[i],s[i]));
		CUDACHECK(cudaEventSynchronize(stop[i]));
		CUDACHECK(cudaEventElapsedTime(&t, start[i], stop[i]));
		elapsedTime = max(elapsedTime, t);
	}

	//free device buffers
	// for (int i = 0; i < nDev; ++i) {
	// 	CUDACHECK(cudaSetDevice(devs[i]));
	// 	for (size_t buff = 0; buff < nbuffs; buff++) {
	// 		CUDACHECK(cudaFree(gradients[i][buff]));
	// 		CUDACHECK(cudaFree(weights[i][buff]));
	// 	}
	// 	CUDACHECK(cudaFree(gradients[i]));
	// 	CUDACHECK(cudaFree(weights[i]));
	// 	CUDACHECK(cudaEventDestroy(stop[i]));
	// 	CUDACHECK(cudaEventDestroy(start[i]));
	// 	CUDACHECK(cudaStreamDestroy(s[i]));
	// }

	for(int i = 0; i < nDev; ++i)
		ncclCommDestroy(comms[i]);

	return elapsedTime;
}

int main(int argc, char* argv[])
{
	if (argc < 3) {
		printf ("Provide four arguments: size scattered/single\n");
		return 0;
	}

	const int size = atoi(argv[1]);  
	const char* mem_type = argv[2];
	printf("num elements %d mem_type %s\n", size, mem_type);

	float elapsedTime;
	if(strcmp(mem_type, "scattered") == 0) {
		const size_t nbuffs = size/1024;
//		const size_t nbuffs = 1;
		size_t buffSizes[nbuffs];
		for (size_t i = 0; i < nbuffs; i++) 
			buffSizes[i] = (size_t)1024;
			//buffSizes[i] = (size_t)size;
		// const size_t nbuffs = 397;
		// const size_t buffSizes[nbuffs] = { 31260672, 524288, 2048, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 1048576, 1048576, 4194304, 4194304, 1048576, 1048576, 2048, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 4096, 1024, 1024, 1024, 1024, 30528, 1024, 1024, 1024 };
		elapsedTime = run_scattered_adam<float>(nbuffs, buffSizes, ncclFloat, false);
	} else {
		// elapsedTime = run_adam<float>(algo, (bool)check_results, size, (float)gradient_factor, ncclFloat, weight_update_time, reduce_time);
	}
	printf ("Elapsed Time: %f\n", elapsedTime);

	return 0;
}
