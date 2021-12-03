include ../common.mk

all: sgd sgd-ar-c sgd-rs-c-ag sgd-fuse-rs-c-ag

sgd: sgd.cpp
	g++ $(COCONET_FLAGS) -g -O0 $< -o $@ -std=c++14

sgd-ar-c.cu: sgd
	./sgd AR_C

sgd-rs-c-ag.cu: sgd
	./sgd RS_C_AG

sgd-fuse-rs-c-ag.cu: sgd
	./sgd fuse\(RS_C_AG\)

sgd-ar-c: sgd-ar-c.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c -o $@.o && $(MPI_CXX) $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

sgd-rs-c-ag: sgd-rs-c-ag.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c -o $@.o && $(MPI_CXX) $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

sgd-fuse-rs-c-ag: sgd-fuse-rs-c-ag.cu

clean:
	rm sgd sgd-ar-c.cu sgd-ar-c sgd-rs-c-ag.cu sgd-rs-c-ag
