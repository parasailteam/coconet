from __future__ import absolute_import
import os
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
print("local_rank", local_rank)
device = torch.device("cuda", local_rank)
print(device)
torch.cuda.set_device(device)
torch.distributed.init_process_group(backend='nccl', init_method='env://')
from apex.optimizers import FusedAdam
from apex import amp
import amp_C
import apex_C

parser = argparse.ArgumentParser(description='benchmark some optimizers')
parser.add_argument('--fp16',action='store_true',default=False)
parser.add_argument('-times',default=1000,type=int)

args = parser.parse_args()
if torch.distributed.get_rank() == 0:
    print("using:", args)

dtype = torch.float16 if args.fp16 else torch.float32
baselines = [
    # ('sgd',torch.optim.SGD),
    ('fusedadam', FusedAdam),
    ('adam',torch.optim.Adam)
]

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

print("<result>")
for name, baseline in baselines:    
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
                optimizer = FusedAdam(params, lr =0.001)
                # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            else:
                optimizer = baseline(params, lr=0.001)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            step_time = 0
            t0 = time.time()
            for t in range(args.times):
                if True:
                    for grad in grads:
                        torch.distributed.all_reduce(grad.data)
                    torch.cuda.synchronize()
                    _t0 = time.time()
                    optimizer.step()
                    torch.cuda.synchronize()
                    _t1 = time.time()
                    
                    step_time += _t1-_t0
                else:
                    take_optimizer_step(device, optimizer, overflow_buf)
            torch.cuda.synchronize()
            t1 = time.time()
            mean_time = (t1-t0)*1000.0 / args.times

            if torch.distributed.get_rank() == 0:
                print(name, size, split_size, (t1-t0)*1000.0, " ms", step_time*1000, " ms", mean_time, " ms", size*4 / mean_time * 1000 / 1000.**3, " GBps")
            
print("</result>")