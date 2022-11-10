#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/tensor_ref.h"
#include "helper.h"

#include <set>
#include <vector>


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

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = cutlass::half_t;                   // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm70;

//cutlass/test/unit/gemm/device/gemm_f16n_f16t_f16t_volta_tensor_op_f16_sm70.cu has some possible Tensor Op Sizes

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N = 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
// // This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<8, 8, 4>;  // <- MMA Op tile M = 8, N = 8, K = 4

// using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 32 
// using ShapeMMAOp = cutlass::gemm::GemmShape<16, 16, 16>;  // <- MMA Op tile M = 8, N = 8, K = 4

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// This code section describes ?
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- this is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 2;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;


template<typename T>
std::vector<std::vector<std::tuple<int, int, int, int>>> getChunkBlocks
  (int rank, size_t matrixSize, int nranks, int* rings, int MATMUL_M, int MATMUL_N,
  const int realChunkCols, int& maxRealChunkRows) 
{
  std::vector<std::vector<std::tuple<int, int, int, int>>> chunkBlocks;

  assert (atoi(getenv ("NCCL_MIN_NCHANNELS")) == atoi(getenv ("NCCL_MAX_NCHANNELS")));
  int nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));
  int nThreads = atoi(getenv("NCCL_NTHREADS"));
  // int nThreadsLL128 = atoi(getenv ("NCCL_LL128_NTHREADS"));
  int channelBuffSize = atoi(getenv("NCCL_BUFFSIZE"));

  const int stepSize = channelBuffSize / (sizeof(T)*NCCL_STEPS);
  const size_t chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
  maxRealChunkRows = 0;

  printf("matrixSize %d nranks * loopSize %d\n", matrixSize, nranks * loopSize);
  for (int userRank = nranks - 1; userRank >= 0; userRank--) {
    chunkBlocks.push_back(std::vector<std::tuple<int, int, int, int>>());
    // int combinedRanks = 1;
    for (int channel = 0; channel < nChannels; channel++) {
      //TODO: following loop only run for once right now.

      for (size_t gridOffset = 0; gridOffset < matrixSize; gridOffset += nranks * loopSize) {
        size_t realChunkSize = min(chunkSize, DIVUP(matrixSize-gridOffset,nranks*nChannels));
        if (matrixSize %3 == 0 && MATMUL_N != 12288) {
          ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T) * 3);
        } else 
        if (matrixSize % 12288 == 0) {
          ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T) * 12);
        }
        else {
          ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T));
        }

        const int realChunkRows = realChunkSize/realChunkCols;
        const int gridOffsetStartRow = gridOffset / MATMUL_N;

        maxRealChunkRows = std::max (maxRealChunkRows, realChunkRows);

        int chunkIdx = rings[channel*nranks + userRank] * nChannels + channel;
        int chunkStartRow = gridOffsetStartRow + chunkIdx / (MATMUL_N / realChunkCols) * realChunkRows;
        int chunkStartCol = chunkIdx % (MATMUL_N / realChunkCols) * realChunkCols;

        int nelem = min(realChunkSize, (matrixSize - (chunkStartRow * MATMUL_N + (MATMUL_M - chunkStartRow) * (MATMUL_N - (MATMUL_N - chunkStartCol)))));
        int chunkRows = min(min(nelem/realChunkCols, realChunkRows), MATMUL_M - chunkStartRow);
        int chunkCols;
        chunkCols = realChunkCols;
        nelem = chunkCols * chunkRows;

        chunkBlocks[chunkBlocks.size() - 1].push_back(std::make_tuple(chunkStartRow, chunkStartCol, chunkRows, chunkCols));
      }
    }
  }

  return chunkBlocks;
}

#define MAX_CHANNELS 80

#include "header.h"
void getCutlassGemm(ncclComm_t comm, Gemm& gemm_op, int M, int N, int K, half* m1, half* m2, half* m1m2, int*& threadBlockToTileMap, int*& tileIdx, int*& tileStatusMap, int*& chunksForTile, int comm_size, int rank)
{
  int ringLength;
  int nChannels;
  int chunkRows;
  int chunkCols = 512;
  assert(N % chunkCols == 0);
  int* rings = new int[MAX_CHANNELS * comm_size];
  getNCCLRing(&comm, rings, ringLength, nChannels);

  std::vector<std::vector<std::tuple<int, int, int, int>>> chunkBlocks = getChunkBlocks<half>(rank, M*N, comm_size, rings, M, N, chunkCols, chunkRows);

    //Overlapped AllReduce + CUTLASS
    int length_m = M;
    int length_n = N;
    int length_k = K;
    cutlass::gemm::GemmCoord problem_size(M, N, K);

    cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a((cutlass::half_t*)m1, LayoutInputA::packed(problem_size.mk()));
    cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b((cutlass::half_t*)m2, LayoutInputA::packed(problem_size.kn()));
    cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c((cutlass::half_t*)m1m2, LayoutInputA::packed(problem_size.mn()));
    cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_d((cutlass::half_t*)m1m2, LayoutInputA::packed(problem_size.mn()));

    // Initialize alpha and beta for dot product computation
    ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
    ElementComputeEpilogue beta = ElementComputeEpilogue(0);

    // Split K dimension into 1 partitions
    int split_k_slices = 1;
            
    //Initialize the memory for thread block to tile map.
    int numTiles = (length_m*length_n)/(ShapeMMAThreadBlock::kMN);
    

    CUDACHECK(cudaMalloc(&tileIdx, sizeof(int)));
    CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));

    CUDACHECK(cudaMalloc(&threadBlockToTileMap, numTiles * 2 * sizeof(int)));

    //An array of integers for each tile to indicate if tile is waiting (0) or finished (1)
    CUDACHECK(cudaMalloc(&tileStatusMap, numTiles * 4 * sizeof(int)));
    CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles * 4 * sizeof(int)));

    //Create an array of tile order.
    // ShapeMMAThreadBlock shape;
    int *tileOrder = new int[numTiles * 2];

    int idx = 0;
    for (int ty = 0; ty < length_m/ShapeMMAThreadBlock::kM; ty++) {
      for (int tx = 0; tx < length_n/ShapeMMAThreadBlock::kN; tx++) {
        tileOrder[idx] = tx;
        tileOrder[idx + 1] = ty;
        idx += 2;
      } 
    }

    std::vector<int> hChunksForTile;
    int maxChunksForTile = 0;

    const int combinedChunks = nChannels;
    
    if (true) {
      idx = 0;
      // int chunk = 0;

      std::set<std::pair<int, int>> chunkTBs;
      std::vector<std::pair<int, int>> tileOrderAsPair;
      std::map<int, std::set<int>> tileToChunks;
          // int tilesForChunk = 0;

      for (auto channelChunks: chunkBlocks) {
        for (int channel = 0; channel < channelChunks.size(); channel++) {
          auto chunk = channelChunks[channel];
          int cy = std::get<0>(chunk);
          int cx = std::get<1>(chunk);
          int m = std::get<2>(chunk);
          int n = std::get<3>(chunk);

          int chunkIndex = cy/chunkRows * N/chunkCols + cx/chunkCols;

          //For a chunk get all tiles required to obtain this chunk
          int startTy = (cy/ ShapeMMAThreadBlock::kM) * ShapeMMAThreadBlock::kM;

          for (int ty = startTy; ty < min(cy + m, length_m); ty += ShapeMMAThreadBlock::kM) {
            for (int tx = cx; tx < min(cx + n, length_n); tx += ShapeMMAThreadBlock::kN) {
              int tileIndex = ty/ShapeMMAThreadBlock::kM * (N/ShapeMMAThreadBlock::kN) + tx/ShapeMMAThreadBlock::kN;
              if (tileToChunks[tileIndex].count(chunkIndex/combinedChunks) == 0) {
                tileToChunks[tileIndex].insert(chunkIndex/combinedChunks);
                // if (rank == 0 && cy >= 7920) {
                //   printf("cy %d cx %d chunkIndex %d\n", cy, cx, chunkIndex);
                //   tilesForChunk++;
                // }
              }

              
              // if (chunkIndex == 0) {
              //   if (rank == 0) 
              //     printf("1199: %d x %d -> %d x %d -> %d\n", 
              //            cy, cx, ty/ShapeMMAThreadBlock::kM, tx/ShapeMMAThreadBlock::kN, tileIndex);
              // }

              if (chunkTBs.count(std::make_pair(ty,tx)) == 0) {
                chunkTBs.insert(std::make_pair(ty,tx));
                // if (rank == 0 && channel == 0) 
                //   printf("%d x %d -> %d x %d -> %d\n", cy, cx, ty/ShapeMMAThreadBlock::kM, tx/ShapeMMAThreadBlock::kN, tileIndex);
                
                tileOrderAsPair.push_back(std::make_pair(tx/ShapeMMAThreadBlock::kN, ty/ShapeMMAThreadBlock::kM));
              }
            }
          }

        }
      }

      // if (rank == 0) {
      //   printf("rank %d tilesForChunk %d\n", rank, tilesForChunk);
      // }

      for (auto v : tileToChunks) {
        maxChunksForTile = std::max(maxChunksForTile, (int)v.second.size());
      }

      hChunksForTile = std::vector<int>(maxChunksForTile * numTiles, 0);

      for (auto it : tileToChunks) {
        int i = 0;
        for (int c : it.second) {
          hChunksForTile[it.first * maxChunksForTile + i] = c;
          i++;
        }
        for (; i < maxChunksForTile; i++) {
          hChunksForTile[it.first * maxChunksForTile + i] = -1;
        }
      }

      int _idx = 0;
      for (int i = 0; i < tileOrderAsPair.size(); i++) {
        tileOrder[_idx] = tileOrderAsPair[i].second; //Swap because x ("m") is row and y ("n") is column.
        tileOrder[_idx+1] = tileOrderAsPair[i].first;

        // printf("%d %d\n", tileOrder[_idx], tileOrder[_idx + 1]);
        _idx += 2;
        idx += 2;
      }    
    }
    
    CUDACHECK(cudaMemcpy(threadBlockToTileMap, tileOrder, numTiles * 2 * sizeof(int), cudaMemcpyHostToDevice));

    CUDACHECK(cudaMalloc(&chunksForTile, hChunksForTile.size() * sizeof(int)));
    CUDACHECK(cudaMemcpy(chunksForTile, &hChunksForTile[0], hChunksForTile.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // delete[] tileOrder;

    typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                        tensor_a,  // <- reference to matrix A on device
                                        tensor_b,  // <- reference to matrix B on device
                                        tensor_c,  // <- reference to matrix C on device
                                        tensor_d,  // <- reference to matrix D on device
                                        maxChunksForTile,
                                        chunksForTile,
                                        tileIdx,
                                        threadBlockToTileMap,
                                        tileStatusMap,
                                        {alpha, beta},          // <- tuple of alpha and beta
                                        split_k_slices};        // <- k-dimension split factor

    // Using the arguments, query for extra workspace required for matrix multiplication computation
    size_t workspace_size = Gemm::get_workspace_size(arguments);

    // Allocate workspace memory
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    // Instantiate CUTLASS kernel depending on templates

    // Check the problem size is supported or not 
    cutlass::Status status = gemm_op.can_implement(arguments);
    CUTLASS_CHECK(status);

    status = gemm_op.initialize(arguments, workspace.get());
    CUTLASS_CHECK(status);

    CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));
    CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles * 4 * sizeof(int)));
}
