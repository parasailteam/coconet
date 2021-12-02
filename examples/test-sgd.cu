#include "header.h"
void pipe(float lr, float* g, float* w, ncclComm_t comm, cudaStream_t stream, size_t SZ){
  NCCLCHECK(AllReduce_pipe(lr, g, w, SZ, ncclFloat32, comm, ncclSum, stream));

  CUDACHECK(cudaStreamSynchronize(stream));
}
bool mpiRef(float* __g, float* __w, float __lr, float* w, size_t SZ, bool dummy=false)
{
  float* __S0;
  __S0 = new float[SZ];
  MPI_Allreduce(__g, __S0, SZ, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  float* __S1;
  __S1 = new float[SZ];
  float* hS1;
  hS1 = new float[SZ];
  CUDACHECK(cudaMemcpy(hS1, w, SZ*sizeof(float), cudaMemcpyDeviceToHost));
  for (size_t i0 = 0; i0 < SZ; i0++) {
    __S1[i0] = (__w[i0] - (__lr * __S0[i0]));
    if (!eqFloat(__S1[i0], hS1[i0])) {
      printf("Mismatch at %ld : ref '%f', computed '%f'\n",i0, __S1[i0], hS1[i0]);
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
  int epochs = 1010;
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  MPI_Barrier(MPI_COMM_WORLD);

  for (auto algo : std::vector<std::string>{"DEFAULT", "Ring", "Tree"}) {
    printf("starting at rank %d for algo %s \n", rank, algo.c_str());

    for (size_t P = 10; P < 29; P++) {
      // Inputs
      size_t SZ = 1 << P;
      float* g;
      CUDACHECK(cudaMalloc(&g, SZ * sizeof(float)));
      cudaMemRandInt(g, SZ);
      float* w;
      CUDACHECK(cudaMalloc(&w, SZ * sizeof(float)));
      cudaMemRandInt(w, SZ);
      float lr;
      lr = 1.0f;

      // Outputs
      float totalTime = 0;
      for(int iter = 0; iter < epochs; iter++) {
        float* __g;;
        if (iter == 0) {
        __g = new float[SZ];;
        CUDACHECK(cudaMemcpy(__g, g, SZ*sizeof(float), cudaMemcpyDeviceToHost));;
        }
        float* __w;;
        if(iter==0){
        __w = new float[SZ];;
        CUDACHECK(cudaMemcpy(__w, w, SZ*sizeof(float), cudaMemcpyDeviceToHost));;
        }
        float __lr;;
        if(iter==0) {
        __lr = __half2float(lr);
        }
        cudaEvent_t startpipe, stoppipe;
        float elapsedTimepipe;
        CUDACHECK(cudaEventCreate(&startpipe));
        CUDACHECK(cudaEventCreate(&stoppipe));
        CUDACHECK(cudaEventRecord(startpipe, 0));
        pipe(lr, g, w, comm, stream, SZ); 
        CUDACHECK(cudaEventRecord(stoppipe, 0));
        CUDACHECK(cudaEventSynchronize(stoppipe));
        CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
        if(iter>10)
        totalTime += elapsedTimepipe;
        if (iter == 0) assert(mpiRef(__g, __w, __lr, w, SZ, false));
      }

      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0) printf("%ld %f ms\n", SZ, totalTime);
    }
  }
  MPI_Finalize();
}
