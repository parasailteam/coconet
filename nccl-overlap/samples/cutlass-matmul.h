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

float cutlassGeMM(const int length_m, const int length_n, const int length_k, int rank, 
                  const std::vector<std::vector<std::tuple<int, int, int, int>>> chunkBlocks) {

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  if (props.major != 7) {
    std::cerr << "Volta Tensor Ops must be run on a machine with compute capability of 70, 72, or 75."
              << std::endl;

    // Return 0 so tests are considered passing if run on unsupported architectures or CUDA Toolkits.
    return 0;
  }

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);

  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());      // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());      // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                               // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                               // reference kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  if (true) {
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(),
      1,
      ElementInputA(4),
      ElementInputA(-4),
      0);  // <- Fill matrix A on host with uniform-distribution random data
  } else {
    cutlass::reference::host::TensorFill(
      tensor_a.host_view(), ElementInputA(__float2half(1)));
  }
  if (true) {
    cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(),
      1,
      ElementInputB(4),
      ElementInputB(-4),
      0);  // <- Fill matrix B on host with uniform-distribution random data
  } else {
    cutlass::reference::host::TensorFill(
      tensor_b.host_view(), ElementInputB(__float2half(1)));
  }
  
  if (true) {
    cutlass::reference::host::TensorFillRandomUniform(
      tensor_c.host_view(),
      1,
      ElementOutput(4),
      ElementOutput(-4),
      0);  // <- Fill matrix C on host with uniform-distribution random data
  } else {
    cutlass::reference::host::TensorFill(
      tensor_c.host_view(), ElementOutput(__float2half(1)));
  }

  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);

  // Split K dimension into 1 partitions
  int split_k_slices = 1;
  
  //Initialize the memory for thread block to tile map.
  int numTiles = (length_m*length_n)/(ShapeMMAThreadBlock::kMN);
  int* threadBlockToTileMap;
  int* tileIdx;
  int* tileStatusMap;

  CUDACHECK(cudaMalloc(&tileIdx, sizeof(int)));
  CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));

  CUDACHECK(cudaMalloc(&threadBlockToTileMap, numTiles * 2 * sizeof(int)));

  //An array of integers for each tile to indicate if tile is waiting (0) or finished (1)
  CUDACHECK(cudaMalloc(&tileStatusMap, numTiles * sizeof(int)));
  CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles * sizeof(int)));

  printf("threadBlockToTileMap %p\n", threadBlockToTileMap);
  //Create an array of tile order.
  ShapeMMAThreadBlock shape;
  int *tileOrder = new int[numTiles * 2];

  int idx = 0;
  for (int ty = 0; ty < length_n/ShapeMMAThreadBlock::kN; ty++) {
    for (int tx = 0; tx < length_m/ShapeMMAThreadBlock::kM; tx++) {
      tileOrder[idx] = tx;
      tileOrder[idx + 1] = ty;
      idx += 2;
    } 
  }

  int chunkM = 64;
  int chunkN = 1536;

  if (true) {
    //Shuffle the ordering to check for correctness

    idx = 0;
    int chunk = 0;

    std::vector<std::pair<int, int>> chunks;

    // for (int cy = 0; cy < length_m; cy += chunkM) {
    //   for (int cx = 0; cx < length_n; cx += chunkN) {
    //     chunks.push_back(std::make_pair(cy, cx));
    //   }
    // }

    // std::random_shuffle (chunks.begin(), chunks.end());
    std::set<std::pair<int, int>> chunkTBs;
    std::vector<std::pair<int, int>> tileOrderAsPair;
    
    int chunkIdx = 0;
    for (auto channelChunks: chunkBlocks) {
      for (auto chunk : channelChunks) {
        int cy = std::get<0>(chunk);
        int cx = std::get<1>(chunk);
        int m = std::get<2>(chunk);
        int n = std::get<3>(chunk);

        //For a chunk get all tiles required to obtain this chunk
        int startTy = (cy/ ShapeMMAThreadBlock::kM) * ShapeMMAThreadBlock::kM;
        // printf("cx x cy: %d x %d\n", cy, cx);

        for (int ty = startTy; ty < min(cy + m, length_m); ty += ShapeMMAThreadBlock::kM) {
          for (int tx = cx; tx < min(cx + n, length_n); tx += ShapeMMAThreadBlock::kN) {
            if (chunkTBs.count(std::make_pair(ty,tx)) == 0) {
              chunkTBs.insert(std::make_pair(ty,tx));
              // printf("%d x %d -> %d x %d\n", cy, cx, ty, tx);
              tileOrderAsPair.push_back(std::make_pair(tx/ShapeMMAThreadBlock::kN, ty/ShapeMMAThreadBlock::kM));
            }
          }
        }
      }
    }

    int _idx = 0;
    for (int i = 0; i < tileOrderAsPair.size(); i++) {
      tileOrder[_idx] = tileOrderAsPair[i].second; //Swap row and column because x ("m") is row and y ("n") is column.
      tileOrder[_idx+1] = tileOrderAsPair[i].first;

      // printf("%d %d\n", tileOrder[_idx], tileOrder[_idx + 1]);
      _idx += 2;
      idx += 2;
    }    
  }

  // printf("size %d %d\n", idx, (length_m*length_n)/ShapeMMAThreadBlock::kMN*2);
  CUDACHECK(cudaMemcpy(threadBlockToTileMap, tileOrder, (length_m*length_n)/ShapeMMAThreadBlock::kMN*2*sizeof(int), cudaMemcpyHostToDevice));
  
  //The remaining is the map array.
  //   CUDACHECK(cudaMemset(threadBlockToTileMap + 1, 0, problem_size.m()*problem_size.k()*sizeof(int)));

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     tensor_c.device_ref(),  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     0,nullptr,
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
  Gemm gemm_op;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  // First run to check results
  for (int part = 0; part < 4; part++) {
    status = gemm_op.runPart(part, 4, 0);
    CUTLASS_CHECK(status);
  
    // Wait for kernels to finish
    CUDACHECK(cudaDeviceSynchronize());
  }


  // Create instantiation for device reference gemm kernel
  cutlass::reference::device::Gemm<ElementInputA,
                                   LayoutInputA,
                                   ElementInputB,
                                   LayoutInputB,
                                   ElementOutput,
                                   LayoutOutput,
                                   ElementComputeEpilogue,
                                   ElementComputeEpilogue>
      gemm_device;
  
  // Launch device reference gemm kernel
  gemm_device(problem_size,
              alpha,
              tensor_a.device_ref(),
              tensor_b.device_ref(),
              beta,
              tensor_c.device_ref(),
              tensor_ref_d.device_ref());

  // Wait for kernels to finish
  cudaDeviceSynchronize();

  // Copy output data from CUTLASS and reference kernel to host for comparison
  tensor_d.sync_host();
  tensor_ref_d.sync_host();

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  bool passed = cutlass::reference::host::TensorEquals(
    tensor_d.host_view(),
    tensor_ref_d.host_view());

  std::cout << (passed ? "Passed" : "Failed") << std::endl;
  
  // Launch initialized CUTLASS kernel
  float totalTime = 0.0f;
  CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));

  for (int iter = 0; iter < 110; iter++) {
    // CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));
    CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles*sizeof(int)));

    double t1 = getCurrentTime();

    status = gemm_op(iter);
    CUTLASS_CHECK(status);

    // Wait for kernels to finish
    CUDACHECK(cudaDeviceSynchronize());
    double t2 = getCurrentTime();
    float elapsedTime2 = (t2 - t1) * 1000.0f;
    if (iter >= 10)
        totalTime += elapsedTime2;
  }

  std::cout << "totalTime with single kernel call " << totalTime << std::endl;

  // // Launch initialized CUTLASS kernel
  totalTime = 0.0f;
  CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));
return 0;
  for (int iter = 0; iter < 110; iter++) {
    
    CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles*sizeof(int)));

    double t1 = getCurrentTime();

    // First run to check results
    for (int i = 0; i < 4; i+=1) {
      status = gemm_op.runPart(i, 4, iter);
      CUTLASS_CHECK(status);
    
      // Wait for kernels to finish
      CUDACHECK(cudaDeviceSynchronize());
    }
    double t2 = getCurrentTime();
    float elapsedTime2 = (t2 - t1) * 1000.0f;
    if (iter >= 10)
        totalTime += elapsedTime2;
  }

  std::cout << "totalTime with parts " << totalTime << std::endl;
  return totalTime;
}