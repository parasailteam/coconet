# CoCoNet

CoCoNet is a DSL and a compiler to co-optimize computation and communication for distributed machine learning workloads.
CoCoNet exposes optimizations across the abstraction boundary of communication and computation routines.
For more details please refer to the paper https://arxiv.org/abs/2105.05720 .

# Code Structure

# Examples


# Installation

## Prerequisites

<b>Linux Installation</b>: We recommend using Ubuntu 20.04 as the Linux OS.

<b>Install Dependencies</b>: Following instructions must be executed on all nodes of the distributed system.

Execute following commands to install dependencies.
```
sudo apt update && sudo apt install gcc linux-headers-$(uname -r) make g++ git python3-dev wget unzip python3-pip cmake openmpi* libopenmpi* libmetis-dev 
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

