include ../common.mk

all: pipeline-parallel pipeline-parallel-ar-c pipeline-parallel-rs-c-ag pipeline-parallel-fuse-rs-c-ag pipeline-parallel-ol-mm-fuse-rs-c-ag

pipeline-parallel: pipeline-parallel.cpp
	g++ $(COCONET_FLAGS) -g -O0 $< -o $@ -std=c++14

pipeline-parallel-ar-p2p-c.cu: pipeline-parallel
	./pipeline-parallel AR_P2P_C

pipeline-parallel-rs-p2p-c-ag.cu: pipeline-parallel
	./pipeline-parallel RS_P2P_C_AG

pipeline-parallel-ol-rs-fuse-p2p-c-ag.cu: pipeline-parallel
	./pipeline-parallel ol_RS_fuse_P2P_C_AG

pipeline-parallel-ar-p2p-c: pipeline-parallel-ar-p2p-c.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ $(GENCODE) -lnccl -lcublas -lcurand -c -o $@.o && $(MPI_CXX) $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

pipeline-parallel-rs-p2p-c-ag: pipeline-parallel-rs-p2p-c-ag.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ $(GENCODE) -lnccl -lcublas -lcurand -c -o $@.o && $(MPI_CXX) $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

pipeline-parallel-ol-rs-fuse-p2p-c-ag: pipeline-parallel-ol-rs-fuse-p2p-c-ag.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c -o $@.o && $(MPI_CXX) $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

clean:
	rm -rf pipeline-parallel pipeline-parallel-ar-p2p-c.cu pipeline-parallel-ar-p2p pipeline-parallel-rs-fuse-p2p-c-ag.cu pipeline-parallel-rs-fuse-p2p-c-ag pipeline-parallel-ol-rs-fuse-p2p-c-ag pipeline-parallel-ol-rs-fuse-p2p-c-ag.cu
