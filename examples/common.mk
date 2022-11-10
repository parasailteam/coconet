COCONET_FLAGS = -I../../src ../../src/codegen.cpp ../../src/dsl.cpp ../../src/pipeline.cpp ../../src/utils.cpp
SCHEDULE_FLAGS = -I../
NCCL_PATH = ../../nccl/
NCCL_BUILD_PATH = $(NCCL_PATH)/build
NCCL_OVERLAP_PATH = ../../nccl-overlap/
NCCL_OVERLAP_BUILD_PATH = $(NCCL_OVERLAP_PATH)/build
MPI_CXX = /usr/bin/mpicxx
GENCODE = "-gencode=arch=compute_70,code=sm_70"
