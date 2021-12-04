from __future__ import absolute_import
import os
from ctypes import *

#DummyAllReduce = CDLL("./libTestAllReduce.so")

os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
# os.environ['MASTER_ADDR'] = os.environ['PHILLY_CONTAINER_IP']
# os.environ['MASTER_PORT'] = str(int(os.environ['PHILLY_CONTAINER_PORT_RANGE_START'])+1)

import time
import torch
import argparse
import numpy as np

from torch.nn.parallel import DistributedDataParallel as DDP

assert (torch.cuda.is_available())
local_rank = int(os.environ['RANK']) % 16 #max(int(os.environ['OMPI_COMM_WORLD_SIZE']), 16)
device = torch.device("cuda", local_rank)
torch.cuda.set_device(device)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
from apex.optimizers import FusedAdam, FusedLAMB
from apex import amp
import apex
import amp_C
import apex_C, sys

parser = argparse.ArgumentParser(description='benchmark some optimizers')
parser.add_argument('--optimizer', default=False,type=str)
parser.add_argument('--fp16',action='store_true',default=False)
parser.add_argument('--times',default=100,type=int)

args = parser.parse_args()
#DummyAllReduce.dummyAllReduce.argtypes = [c_ulonglong, c_ulonglong, c_int]
dtype = torch.float16 if args.fp16 else torch.float32
#LP_c_char = POINTER(c_char)
#LP_LP_c_char = POINTER(LP_c_char)
#DummyAllReduce.initTestAllReduce.argtypes = [c_int, LP_LP_c_char]

#p = (LP_c_char*len(sys.argv))()
#for i, arg in enumerate(sys.argv):  # not sys.argv, but argv!!!
#  enc_arg = arg.encode('utf-8')
#  p[i] = create_string_buffer(enc_arg)

#na = cast(p, LP_LP_c_char)
argc = len(sys.argv)
#DummyAllReduce.initTestAllReduce(argc, na)

# baselines = [
#     # ('sgd',torch.optim.SGD),
#     ('FusedAdam', FusedAdam),
#     ('FusedLAMB',FusedLAMB),
#     # ('PyTorchAdam',torch.optim.Adam),
# ]
baselines = {'FusedAdam' : FusedAdam, 'FusedLAMB':FusedLAMB}

assert args.optimizer in [o for o in baselines.keys()]

overflow_buf = torch.cuda.IntTensor([0]).to(device)

def take_optimizer_step(device, optimizer, overflow_buf):
    master_grads = [p.grad for p in amp.master_params(optimizer) if p.grad is not None]
    #assert len(master_grads) == len(list(amp.master_params(optimizer)))
    flat_grad_size = sum(p.numel() for p in master_grads)
    flat_raw = torch.empty(flat_grad_size, device=device, dtype=torch.float32)
    #flat_raw = optimizer.param_groups[0]['params'][0].grad
    allreduced_views = apex_C.unflatten(flat_raw, master_grads)
    overflow_buf.zero_()
    amp_C.multi_tensor_scale(65536,
        overflow_buf,
        [master_grads, allreduced_views],
        1)
    torch.distributed.all_reduce(flat_raw)
    amp_C.multi_tensor_scale(65536,
        overflow_buf,
        [allreduced_views, master_grads],
        1.)
    if overflow_buf.item() > 0:
        raise Exception("oops")
    t0 = time.time()
    optimizer.step()
    t1 = time.time()

    #torch.distributed.all_reduce(optimizer.param_groups[0]['params'][0].grad)
    
    return (t1-t0)

if os.environ['RANK'] == 0:
    print("<result>")
for name, baseline in [(args.optimizer, baselines[args.optimizer])]:
    for i in range(10, 31):
        size = 2**i
        flat_params = torch.zeros(size, dtype=dtype).to(device)
        flat_grads  = torch.zeros(size, dtype=dtype).to(device)
        for split_size in [size]:
            params = torch.split(flat_params, split_size)
            grads = torch.split(flat_grads, split_size)
            for param, grad in zip(params, grads):
                param.grad = grad
            if (baseline == FusedAdam):
                optimizer = FusedAdam(params, lr=0.001)
                # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            if (baseline == FusedLAMB):
                optimizer = FusedLAMB(params, lr=0.001)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            step_time = 0
            t0 = time.time()
            epochs = args.times
            optimizerCTime = 0
            array_type = c_float * size
            tmpInArray = grad.data_ptr()
            isfp16 = 1 if args.fp16 else 0
            for t in range(epochs):
                if True:
                    for grad in grads:
                        torch.distributed.all_reduce(grad.data)
                        #DummyAllReduce.dummyAllReduce(c_ulonglong(tmpInArray), c_ulonglong(size), c_int(isfp16))
                    torch.cuda.synchronize()
                    _t0 = time.time()
                    optimizer.step()
                    torch.cuda.synchronize()
                    _t1 = time.time()
                    #optimizerCTime += optimizer.totalTime
                    
                    step_time += _t1-_t0
                else:
                    pass#take_optimizer_step(device, optimizer, overflow_buf)
            torch.cuda.synchronize()
            t1 = time.time()
            mean_time = (t1-t0)*1000.0 / epochs

            if torch.distributed.get_rank() == 0:
                print(name + ": {SZ: %ld, Epochs: %d, Total: %f, AllReduce: %f, %s: %f, %s-CTime: %f}"%(size, epochs, (t1-t0)*1000.0, (t1-t0)*1000.0 - step_time*1000, name, step_time*1000, name, optimizerCTime*1000))
                #print(name, size, split_size, (t1-t0)*1000.0, " ms", step_time*1000, " ms", mean_time, " ms", size*4 / mean_time * 1000 / 1000.**3, " GBps")

if os.environ['RANK'] == 0:      
    print("</result>")
