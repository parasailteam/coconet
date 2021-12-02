#include "header.h"
void pipe(float lr, float beta1, float beta2, float* g, float* w, float* m, float* v, ncclComm_t comm, cudaStream_t stream, size_t SZ){
  NCCLCHECK(AllReduce_pipe(lr, beta1, beta2, g, w, m, v, SZ, ncclFloat32, comm, ncclSum, stream));

  CUDACHECK(cudaStreamSynchronize(stream));
}
bool mpiRef(float* __g, float* __w, float* __m, float* __v, float __lr, float __beta1, float __beta2, float* w, float* m, float* v, size_t SZ, bool dummy=false)
{
  float* __S0;
  __S0 = new float[SZ];
  MPI_Allreduce(__g, __S0, SZ, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float* __S2;
  __S2 = new float[SZ];
  for (size_t i0 = 0; i0 < SZ; i0++) {
    __S2[i0] = ((__beta2 * __v[i0]) + ((1 - __beta2) * (__S0[i0] * __S0[i0])));
  }
  float* __S4;
  __S4 = new float[SZ];
  for (size_t i0 = 0; i0 < SZ; i0++) {
    __S4[i0] = (__S2[i0] / __beta2);
  }
  float* __S1;
  __S1 = new float[SZ];
  for (size_t i0 = 0; i0 < SZ; i0++) {
    __S1[i0] = ((__beta1 * __m[i0]) + ((1 - __beta1) * __S0[i0]));
  }
  float* __S3;
  __S3 = new float[SZ];
  for (size_t i0 = 0; i0 < SZ; i0++) {
    __S3[i0] = (__S1[i0] / __beta1);
  }
  float* __S5;
  __S5 = new float[SZ];
  float* hS5;
  hS5 = new float[SZ];
  CUDACHECK(cudaMemcpy(hS5, w, SZ*sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i0 = 0; i0 < SZ; i0++) {
    __S5[i0] = (__w[i0] - ((__lr * __S3[i0]) / __S4[i0]));
    if (!eqFloat(__S5[i0], hS5[i0])) {
      printf("Mismatch at %ld : ref '%f', computed '%f'\n",i0, __S5[i0], hS5[i0]);
      return false;
    }
  }
  return true;
}
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
  int epochs = 1010;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  if (rank == 0)
    printf("adam\n");
  
    // printf("starting at rank %d for algo %s \n", rank, algo.c_str());
    ncclCommInitRank(&comm, comm_size, id, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    printf("<results>\n");
    for (int  P = 10; P < 31; P++) {
      size_t SZ = 1UL << P;
      // Inputs
      float* g;
      CUDACHECK(cudaMalloc(&g, SZ * sizeof(float)));
      memset_value(g, 1.0f, SZ);
      float* w;
      CUDACHECK(cudaMalloc(&w, SZ * sizeof(float)));
      memset_value(w, 0.0f, SZ);
      float* m;
      CUDACHECK(cudaMalloc(&m, SZ * sizeof(float)));
      memset_value(m, 0.0f, SZ);
      float* v;
      CUDACHECK(cudaMalloc(&v, SZ * sizeof(float)));
      memset_value(v, 0.0f, SZ);
      float lr;
      lr = 1.0f;
      float beta1;
      beta1 = 0.5f;
      float beta2;
      beta2 = 0.5f;

      // Outputs
      float totalTime = 0;
      for(int iter = 0; iter < epochs; iter++) {
        // printf("iter %d\n", iter);
        float* __g;;
        if (iter == 0) {
          __g = new float[SZ];;
          CUDACHECK(cudaMemcpy(__g, g, SZ*sizeof(float), cudaMemcpyDeviceToHost));;
        }
        float* __w;;
        if (iter == 0) {
          __w = new float[SZ];;
          CUDACHECK(cudaMemcpy(__w, w, SZ*sizeof(float), cudaMemcpyDeviceToHost));;
        }
        float* __m;;
        if(iter == 0) {
          __m = new float[SZ];;
          CUDACHECK(cudaMemcpy(__m, m, SZ*sizeof(float), cudaMemcpyDeviceToHost));;
        }
        float* __v;;
        if(iter == 0) {
          __v = new float[SZ];;
          CUDACHECK(cudaMemcpy(__v, v, SZ*sizeof(float), cudaMemcpyDeviceToHost));;
        }
        float __lr;;
        if (iter == 0)
        __lr = __half2float(lr);
        float __beta1;;
        if(iter == 0)
        __beta1 = __half2float(beta1);
        float __beta2;;
        if (iter == 0)
        __beta2 = __half2float(beta2);
        cudaEvent_t startpipe, stoppipe;
        float elapsedTimepipe;
        CUDACHECK(cudaEventCreate(&startpipe));
        CUDACHECK(cudaEventCreate(&stoppipe));
        CUDACHECK(cudaEventRecord(startpipe, 0));
        pipe(lr, beta1, beta2, g, w, m, v, comm, stream,SZ); 
        CUDACHECK(cudaEventRecord(stoppipe, 0));
        CUDACHECK(cudaEventSynchronize(stoppipe));
        CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
        if (iter > 10)
        totalTime += elapsedTimepipe;
        if (iter == 0) assert(mpiRef(__g, __w, __m, __v, __lr, __beta1, __beta2, w, m, v, false));
      }
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
      printf("%s %ld %f ms\n", getenv ("NCCL_ALGO"), SZ, totalTime);

      cudaFree(g);
      cudaFree(w);
      cudaFree(m);
      cudaFree(v);
    }
    if(rank == 0)
    printf("</results>\n");
    ncclCommDestroy(comm);

  MPI_Finalize();
}
