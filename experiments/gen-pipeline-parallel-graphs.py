import subprocess
import sys
import os
import math
import random
import datetime
import shutil
import re
import json

FULL_PERF_EVAL = "1000"
epochs = 1

if len(sys.argv) < 2:
    print("Results directory not specified")
    sys.exit(0)

def slurp(filepath):
    f = open(filepath, "r")
    s = f.read()
    f.close()

    return s

nccl_path = os.path.abspath("../nccl")
resultsDir = os.path.abspath(sys.argv[1])
assert "NPROC" in os.environ, "Set NPROC to number of processes"
nranks = os.environ.get("NPROC")

resultsDir = os.path.abspath(resultsDir)
if not os.path.exists(resultsDir):
    print ('Results directory "%s" do not exist.'%(resultsDir))
    
parent_dir = os.getcwd()
nccl_path = os.path.join(parent_dir, nccl_path)

def execute_command(c):
    s, o = subprocess.getstatusoutput(c)
    if s != 0:
        raise Exception("Command '%s' unsuccessful:\n"%c +o)
    return (s, o)

results = {"pipeline-parallel-ar-p2p-c": [],
           "pipeline-parallel-rs-p2p-c-ag": [],
           "pipeline-parallel-ol-rs-fuse-p2p-c-ag": []}

for appdir in os.listdir(resultsDir):
    command = slurp(os.path.join(resultsDir, appdir, "json.json"))
    binary = command[command.find("pipeline-parallel-"):].split(" ")[0].strip()
    resultstxt = slurp(os.path.join(resultsDir, appdir, "stdout.txt"))
    dicts = re.findall(r'{.+}', resultstxt)
    dicts = [re.sub(r'([a-zA-Z_]+\d*)',r'"\1"', s) for s in dicts]
    print (dicts)
    resultsjson = [json.loads(s) for s in dicts]
    results[binary] = resultsjson

print(results)

binaries = ["pipeline-parallel-ar-p2p-c", "pipeline-parallel-rs-p2p-c-ag", "pipeline-parallel-ol-rs-fuse-p2p-c-ag"]

rows_H = []

for i in [0, 1]:
    binaryResult = results[binaries[0]]
    B_8_results = binaryResult[i]
    row = [(i+1)*8, B_8_results["AllReduce"], B_8_results["matMul0"], B_8_results["binOpFunc0"], B_8_results["Total"]]
    binaryResult = results[binaries[1]]
    B_8_results = binaryResult[i]
    row += [B_8_results["binOpFunc0"], B_8_results["ReduceScatter"], B_8_results["AllGather"], B_8_results["Total"]]
    binaryResult = results[binaries[2]]
    B_8_results = binaryResult[i]
    row += [B_8_results["overlap"]]

    rows_H += [row]

print (rows_H)
rows_4H = list(rows_H)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys 
from matplotlib import ticker
from functools import reduce

def remove_chunk_from_string(s):
    return s[s.find(')')+1:].strip()

def batch_size_from_matmul(matmul):
    if str(8*1024) in matmul:
        return 8
    elif str(16*1024) in matmul:
        return 16
    elif str(32*1024) in matmul:
        return 32
    elif str(64*1024) in matmul:
        return 64

x = []
small_y = [[],[],[]] #relative bandwidth
small_cublas_y = []
small_allreduce_y = []
small_compute_y = [[], []]
small_reducescatter_y = []
small_allgather_y = []

big_y = [[], [], []]
big_cublas_y = []
big_allreduce_y = []
big_compute_y = [[], []]
big_reducescatter_y = []
big_allgather_y = []

gshard_speedup = []
coconet_speedup = []
megatron_speedup = []
coconet_baseline_speedup = []

megatron_speedup_big = [1.05, 1.05, 1.06, 1.06]
megatron_speedup_small = [1.06, 1.07, 1.08, 1.09]

for row in rows_H:
    data = row
    
    if row[0] == "":
        continue
    
    if True: # skip header and 512 size
        x.append(data[0])
        baseline = float(data[4])
        matmul_time = float(data[2])
        allreduce = float(data[1])
        small_allreduce_y.append(allreduce/baseline)
        small_cublas_y.append(matmul_time/baseline)
        small_compute_y[0].append(1 - (small_allreduce_y[-1] + small_cublas_y[-1]))
        small_y[0].append(1/ (baseline / baseline ))

        gshard = float(data[8])
        ag = float(data[7])
        rs = float(data[6])
        sliced_compute = float(data[5])
        
        diff = gshard - ag - rs - sliced_compute - matmul_time
        small_allgather_y += [(ag+diff/2)/baseline]
        small_reducescatter_y += [(rs+diff/2)/baseline]
        small_compute_y[1] += [(sliced_compute)/baseline]
        small_y[1].append( 1/((baseline * 1.05)/ gshard))
        small_y[2].append( 1/((baseline * 1.05) / row[9]))
        gshard_speedup.append(baseline/gshard* 1.05)
        coconet_speedup.append(1/small_y[2][-1])
        coconet_baseline_speedup.append(1.05)
        # bwy[1].append(data[6])
        # bwy[2].append(data[7])
        # bwy[3].append(data[8]) 
    
    if True: # skip header and 512 size
        baseline = float(data[4])
        matmul_time = float(data[2])
        small_allreduce_y.append(float(data[1])/baseline)
        small_cublas_y.append(matmul_time/baseline)
        small_compute_y[0].append(1 - (small_allreduce_y[-1] + small_cublas_y[-1]))
        small_y[0].append(1/ (baseline / float(data[4])))
        gshard = float(data[8])
        ag = float(data[7])
        rs = float(data[6])
        sliced_compute = float(data[5])
        diff = gshard - ag - rs - sliced_compute - matmul_time
        small_allgather_y += [(ag+diff/2)/baseline]
        small_reducescatter_y += [(rs+diff/2)/baseline]
        small_compute_y[1] += [(sliced_compute)/baseline]
        small_y[1].append( 1/((baseline * 1.05) / gshard))
        small_y[2].append( 1/((baseline * 1.05) / row[9]))
        gshard_speedup.append(1/small_y[1][-1])
        coconet_speedup.append(1/small_y[2][-1])
        coconet_baseline_speedup.append(1.05)

def autolabel(rects, values):
    """
    Attach a text label above each bar displaying its height
    """
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., rect.get_y() + height,
                r"{: .2f}$\times$".format(values[i]),
                ha='center', va='bottom', fontsize=10, rotation=90)

print(small_y[2])
#### Smaller Matmul
small_compute_y = [np.array(x) for x in small_compute_y]
x = x + x
xx = np.arange(len(x))
barwidth = 0.2
fig, ax = plt.subplots()
ax.minorticks_on()

colors={"AR":'#093D87', "MM":'#2171B6', "C":'#cde6c7', "AG":'#6FBBE2', "RS":'#3C7255', "coconet": '#F58E35'}

ax.bar(xx+0*barwidth, small_allreduce_y, color = colors["AR"], edgecolor='white', width=barwidth)
ax.bar(xx+0*barwidth, small_cublas_y, bottom=small_allreduce_y, color = colors["MM"], edgecolor='white', width=barwidth)
ax.bar(xx+0*barwidth, small_compute_y[0], bottom=np.add(small_cublas_y, small_allreduce_y).tolist(), color = colors["C"], edgecolor='white', width=barwidth)

ax.bar(xx+1*barwidth, small_allreduce_y, color = colors["AR"], edgecolor='white', width=barwidth, label='AR')
ax.bar(xx+1*barwidth, small_cublas_y, bottom=small_allreduce_y, color = colors["MM"], edgecolor='white', width=barwidth, label='MM')
coconet_baseline_rects = ax.bar(xx+1*barwidth, small_compute_y[0]/1.5, bottom=np.add(small_cublas_y, small_allreduce_y).tolist(), color = colors["C"], edgecolor='white', width=barwidth, label='C')

ax.bar(xx+2*barwidth, small_reducescatter_y, color = colors["RS"], width=barwidth, edgecolor='white', label='RS')
ax.bar(xx+2*barwidth, small_allgather_y, bottom=small_reducescatter_y, color = colors["AG"], width=barwidth, edgecolor='white', label='AG')
ax.bar(xx+2*barwidth, small_cublas_y, bottom=np.add(small_reducescatter_y, small_allgather_y).tolist(), color = colors["MM"], width=barwidth, edgecolor='white')

gshard_rects = ax.bar(xx+2*barwidth, small_compute_y[1], bottom=np.add(np.add(small_reducescatter_y, small_allgather_y), small_cublas_y).tolist(), color = colors["C"], width=barwidth, edgecolor='white')
coconet_rects = ax.bar(xx+3*barwidth, small_y[2], color = colors["coconet"], width=barwidth, label='Overlap+Fuse', edgecolor='white')


autolabel(coconet_baseline_rects, coconet_baseline_speedup)
autolabel(gshard_rects, gshard_speedup)
autolabel(coconet_rects, coconet_speedup)

rects_locs = xx.tolist() + (xx + 1*barwidth + 0.0001).tolist() + (xx + 2*barwidth).tolist() + (xx + 3*barwidth).tolist()

ax.set_xticks(rects_locs, minor = True)
new_ticks = reduce(lambda x, y: x + y, map(lambda x: [x] * 4, ["MegatronLM", "MM-AR-C", "GShard-Eq", "CoCoNet"]))
ax.xaxis.set_minor_formatter(ticker.FixedFormatter(new_ticks))  #add the custom ticks
ax.tick_params(axis='x', which='major', pad=60)
ax.tick_params(axis='x', which='both',labelsize=12)

plt.ylabel('Times normalized to\n MegatronLM', fontsize=12)
plt.xticks([r + barwidth for r in range(len(xx))], ['B=%d'%val  for val in x])
ax.set_ylim([0.2,1.2])
plt.legend(loc='upper left', fontsize='large',bbox_to_anchor=(-0.1, 1.26),ncol=6,columnspacing=1,handlelength=1.7)
ax.grid(axis='y')
plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, ha='right')
ax.text(0,-0.52, "[B, S, H/16] x [H/16, H]", fontsize=12)
ax.text(1.9,-0.52, "[B, S, 4*H/16] x [4*H/16, H]", fontsize=12)

fig = plt.gcf()
fig.subplots_adjust(bottom=0.38)
fig.set_size_inches(6.8, 3.8)

fig.savefig("Figure 11.pdf",bbox_inches=0,pad_inches=0)