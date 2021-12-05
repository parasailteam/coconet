include ../common.mk

all: lamb lamb-ar-c lamb-rs-c-ag lamb-fuse-rs-c-ag

lamb: lamb.cpp
	g++ $(COCONET_FLAGS) $< -g -O0 -o $@ -std=c++14

lamb-ar-c.cu: lamb
	./lamb AR_C

lamb-rs-c-ag.cu: lamb
	./lamb RS_C_AG

lamb-fuse-rs-c-ag.cu: lamb
	./lamb fuse_RS_C_AG

lamb-ar-c: lamb-ar-c.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c -o $@.o && mpicxx $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

lamb-rs-c-ag: lamb-rs-c-ag.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c -o $@.o && mpicxx $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

lamb-fuse-rs-c-ag: lamb-fuse-rs-c-ag.cu
	nvcc $< -std=c++11 -Xcompiler -fopenmp -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/openmpi -I.. -I/usr/local/cuda/include/ -I../../ -I$(NCCL_BUILD_PATH)/include -L$(NCCL_BUILD_PATH)/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c -o $@.o && mpicxx $@.o -L$(NCCL_BUILD_PATH)/lib -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o $@

clean:
	rm -rf lamb lamb-ar-c.cu lamb-ar-c lamb-rs-c-ag.cu lamb-rs-c-ag lamb-fuse-rs-c-ag.cu lamb-fuse-rs-c-ag
