# CoCoNet

CoCoNet is a DSL and a compiler to co-optimize computation and communication for distributed machine learning workloads.
CoCoNet exposes optimizations across the abstraction boundary of communication and computation routines.
For more details please refer to the paper https://arxiv.org/abs/2105.05720 .

# Directory Structure
* `examples/` 
    * `sgd/` contains Parameter Update using Stochastic Gradient Descent (SGD) optimizer
    * `adam/` contains Parameter Update using Adam Optimizer
    * `lamb/` contains Parameter Update using LAMB Optimizer
* `src/` contains the source of code CoCoNet
* `nccl-src/` contains modified NCCL source cloned from `https://github.com/NVIDIA/nccl/`

# CoCoNet Example
The key parts of a distributed machine learning program can be written and optimized in CoCoNet with only few lines of code. This section uses the SGD example in `examples/sgd/sgd.cpp`. 
First we declare the input tensors in our program.
Each tensor can have one of three layouts:
1. <i>Local</i>: A Local tensor is present on all nodes of distributed system and each tensor contains different elements.
2. <i>Replicated</i>: A Replicated tensor is present on all nodes of distributed system and but tensor contains same elements on nodes.
3. <i>Sliced</i>: A Sliced tensor is divided on all nodes of distributed system and i<sup>th</sup> node contains i<sup>th</sup> part of the tensor.

```
Variable N(Int32, "N");    //Length of g and w
Variable lr(Float32, "lr");  //Learning rate
Tensor g(Float32, N, Local, "g"); //Gradient
Tensor w(Float32, N, Replicated, "w"); //Weights (parameters)
```

Now we perform the operations on the tensors to produce new tensors.
CoCoNet supports communication operations and computation operations.
In SGD, first do an AllReduce and then parameter update.
```
Stage g1 = AllReduce(Summation, g);
Stage w1 = Update(w, w - lr * g1);
```

Finally create the program and generate code.
```
Pipeline pipeline({g,w,lr}, {w1});

pipeline.codegen("sgd-ar-c.cu");
```

# Prerequisites

<b>Linux Installation</b>: We recommend using Ubuntu 20.04 as the Linux OS.

<b>Install Dependencies</b>: Following instructions must be executed on all nodes of the distributed system.

Execute following commands to install dependencies.
```
sudo apt update && sudo apt install gcc linux-headers-$(uname -r) make g++ git python3-dev wget unzip python3-pip cmake openmpi* libopenmpi* *glog*
```

<b>Install CUDA</b>: In our experiments we used CUDA 11.3 on Ubuntu 20.04. We cannot use CUDA 11.5 because currently PyTorch only supports upto 11.3. CUDA 11.3 toolkit can be downloaded from https://developer.nvidia.com/cuda-11.3.0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=runfile_local
While installing CUDA, install CUDA in /usr/local/cuda and install the CUDA samples in the home directory. 

<b>Check CUDA Installation</b>: To check CUDA installation, go to CUDA samples installed in your home directory and execute following commands:
```
cd ~/NVIDIA_CUDA-11.3_Samples/1_Utilities/deviceQuery
make
./deviceQuery
```

Executing this deviceQuery command will show the information about GPUs on the node. If there is any error then either CUDA device is not present or CUDA driver is not installed correctly.

Set NVCC Path and CUDA libraries path: We assume that nvcc is present in /usr/local/cuda/bin/nvcc. Please make sure that this is a valid path and this nvcc is from CUDA 11.3 by using nvcc --version. Then export this in your PATH variable.
```
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
```

<b>Install Anaconda</b> We will use Anaconda to download and build PyTorch. You can download Anaconda from linux by going to https://www.anaconda.com/products/individual#linux . We recommend the bash file available at https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh .
After download install Anaconda by
```
sh ./Anaconda3-2021.11-Linux-x86_64.sh
```
Restart the shell and activate the Anaconda environment by
```
source /path/to/conda/bin/activate 
```
This activation is needed after every login to shell.
Anaconda installs mpich but we need to remove it.
```
conda remove mpich 
```
Install matplotlib by
```
conda install matplotlib
```

<b>Install PyTorch</b> Install PyTorch using pip for CUDA 11.3. The instructions are available at https://pytorch.org/get-started/locally/
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Verify PyTorch and CUDA Installation by executing following text in `python`:
```
import torch
torch.cuda.is_available()
```
This should return `True`, otherwise please check your PyTorch and CUDA installations.

<b>Install NVIDIA Apex Optimizer</b> NVIDIA Apex (https://github.com/NVIDIA/apex) is one of the baselines. Install it by following the Quick Start guide in README.md of `apex` repository. Following commands are copied from there.

```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

# Building Examples

`examples/` directory contains several examples. To run an example, say `sgd`, do
```
cd examples/sgd
make 
```
This will generate three programs for SGD: 
1. `sgd-ar-c`: that performs an AllReduce and a weight update.
2. `sgd-rs-c-ag`: that performs a ReduceScatter, perform <i>sliced</i> weight updates, and then an AllGather to gather all updated weights.
3. `sgd-fuse-rs-c-ag`: that fuses all operations in previous program in a single FusedAllReduce call.

# Experiments

Running experiments requires setting two environment variables:
* `NPROC` to specify the number of processes to invoke
* `MPI_ARGS` to specify the any arguments to MPI. Testing script uses `torch.distributed` that requires specifying `MASTER_ADDR` and `MASTER_PORT` arguments. A hostfile can also be passed through `MPI_ARGS`.

<b>Data Parallel Experiments</b>
To run experiments invoke `experiments/data-parallel-exp.py`. The script requires an argument: a directory to store results. 
For example, to run experiments with 4 processes, with MASTER_ADDR to 127.0.0.1, MASTER_PORT to 10000, and store results to `experiments/results`.

```
cd experiments
export NPROC=4
export MPI_ARGS="-x MASTER_ADDR=127.0.0.1 -x MASTER_PORT=10000"
python data-parallel-exp.py results/
```

Generate graphs in PDF format by running `experiments/gen-data-parallel-graphs.py` and provide the results directory.

```
python gen-data-parallel-graphs.py results/
```

Graphs with following names are stored in `experiments` directory:
* `Figure10a.pdf`: Shows the results for Adam optimizer
* `Figure10b.pdf`: Shows the results for LAMB optimizer

<b>Model Parallel Experiments</b>
To run experiments invoke `experiments/model-parallel-exp.py`. The script requires an argument: a directory to store results. 
For example, to run experiments with 4 processes, with MASTER_ADDR to 127.0.0.1, MASTER_PORT to 10000, and store results to `experiments/results`.

```
cd experiments
export NPROC=4
export MPI_ARGS="-x MASTER_ADDR=127.0.0.1 -x MASTER_PORT=10000"
python model-parallel-exp.py results/
```

Generate graphs in PDF format by running `experiments/gen-model-parallel-graphs.py` and provide the results directory.

```
python gen-model-parallel-graphs.py results/
```

Graphs with `Figure11.pdf` is stored in `experiments` directory.

<b>Pipeline Parallel Experiments</b>
To run experiments invoke `experiments/pipeline-parallel-exp.py`. The script requires an argument: a directory to store results. 
For example, to run experiments with 4 processes, with MASTER_ADDR to 127.0.0.1, MASTER_PORT to 10000, and store results to `experiments/results`.

```
cd experiments
export NPROC=4
export MPI_ARGS="-x MASTER_ADDR=127.0.0.1 -x MASTER_PORT=10000"
python pipeline-parallel-exp.py results/
```

Generate graphs in PDF format by running `experiments/gen-pipeline-parallel-graphs.py` and provide the results directory.

```
python gen-pipeline-parallel-graphs.py results/
```

Graphs with `Figure12.pdf` is stored in `experiments` directory.
