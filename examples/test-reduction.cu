#include "header.h"
void pipe(float lr, float* g, float* w, ncclComm_t comm, cudaStream_t stream){
  NCCLCHECK(AllReduce_pipe(g, w, 8192, ncclFloat32, comm, ncclSum, stream));

  CUDACHECK(cudaStreamSynchronize(stream));
}
bool mpiRef(float* __g, float* __w, float __lr, float* w, bool dummy=false)
{
  float* __S0;
  __S0 = new float[8192UL];
  MPI_Allreduce(__g, __S0, 8192, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float* __S1;
  __S1 = new float[8192UL];
  for (size_t i0 = 0; i0 < 8192; i0++) {
    __S1[i0] = (__w[i0] + __S0[i0]);
  }
  float __S2;
  for (size_t i0 = 0; i0 < 8192; i0++) {
    __S2 = __S2+__S1[i0];
  }
  float* __S3;
  __S3 = new float[8192UL];
  float* hS3;
  hS3 = new float[8192UL];
  CUDACHECK(cudaMemcpy(hS3, w, 8192*sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i0 = 0; i0 < 8192; i0++) {
    __S3[i0] = (__w[i0] - (__S2 * __S0[i0]));
    if (!eqFloat(__S3[i0], hS3[i0])) {
      printf("Mismatch at %ld : ref '%f', computed '%f' __w[i0]  %f __S2[0]  %f __S0[i0] %f __S1[i0] %f\n",i0, __S3[i0], hS3[i0], __w[i0], __S2, __S0[i0], __S1[i0]);
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
  ncclCommInitRank(&comm, comm_size, id, rank);
  int epochs = 1;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  MPI_Barrier(MPI_COMM_WORLD);

  // Inputs
  float* g;
  CUDACHECK(cudaMalloc(&g, 8192 * sizeof(float)));
  memset_value(g, 1.0f, 8192);
  float* w;
  CUDACHECK(cudaMalloc(&w, 8192 * sizeof(float)));
  memset_value(w, 0.0f, 8192);
  float lr;
  lr = 1.0f;

  // Outputs
  float totalTime = 0;
  for(int iter = 0; iter < epochs; iter++) {
    float* __g;;
    __g = new float[8192UL];;
    CUDACHECK(cudaMemcpy(__g, g, 8192*sizeof(float), cudaMemcpyDeviceToHost));;
    float* __w;;
    __w = new float[8192UL];;
    CUDACHECK(cudaMemcpy(__w, w, 8192*sizeof(float), cudaMemcpyDeviceToHost));;
    float __lr;;
    
    cudaEvent_t startpipe, stoppipe;
    float elapsedTimepipe;
    CUDACHECK(cudaEventCreate(&startpipe));
    CUDACHECK(cudaEventCreate(&stoppipe));
    CUDACHECK(cudaEventRecord(startpipe, 0));
    pipe(lr, g, w, comm, stream); 
    CUDACHECK(cudaEventRecord(stoppipe, 0));
    CUDACHECK(cudaEventSynchronize(stoppipe));
    CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
    totalTime += elapsedTimepipe;
    if (iter == 0) assert(mpiRef(__g, __w, __lr, w, false));
  }
  MPI_Finalize();
  printf("Total Time %f ms\n",totalTime);
}