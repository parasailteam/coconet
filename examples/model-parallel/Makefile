include ../common.mk

all: model-parallel model-parallel-ar-c model-parallel-rs-c-ag model-parallel-fuse-rs-c-ag model-parallel-ol-mm-fuse-rs-c-ag

model-parallel: model-parallel.cpp
	g++ $(COCONET_FLAGS) -g -O0 $< -o $@ -std=c++14

model-parallel-mm-ar-c.cu: model-parallel
	./model-parallel MM_AR_C

model-parallel-mm-rs-c-ag.cu: model-parallel
	./model-parallel MM_RS_C_AG

model-parallel-ol-mm-fuse-rs-c-ag.cu: model-parallel
	./model-parallel ol_MM_fuse_RS_C_AG

model-parallel-mm-ar-c: model-parallel-mm-ar-c.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ $(GENCODE) -lnccl -lcublas -lcurand -c -o $@.o && $(MPI_CXX) $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

model-parallel-mm-rs-c-ag: model-parallel-mm-rs-c-ag.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ $(GENCODE) -lnccl -lcublas -lcurand -c -o $@.o && $(MPI_CXX) $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

model-parallel-ol-mm-fuse-rs-c-ag: model-parallel-ol-mm-fuse-rs-c-ag.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c -o $@.o && $(MPI_CXX) $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

clean:
	rm -rf model-parallel model-parallel-ar-c.cu model-parallel-ar-c model-parallel-rs-c-ag.cu model-parallel-rs-c-ag model-parallel-ol-mm-fuse-rs-c-ag model-parallel-ol-mm-fuse-rs-c-ag.cu
