import os
import re
import json
import ast 
import csv
import sys
import shutil
 # ["allreduce-lambf16", "reducescatter-lamb-allgatherf16", "test-lambf16"] + \
 # ["allreduce-adamf16", "reducescatter-adam-allgatherf16", "test-adamf16"] +\
all_binaries = ["adam-ar-c", "adam-rs-c-ag", "adam-fuse-rs-c-ag"] + \
    ["lamb-ar-c", "lamb-rs-c-ag", "lamb-fuse-rs-c-ag"] + \
    ["python3 optimbench.py --optimizer FusedLAMB --fp16", "python3 optimbench.py --optimizer FusedLAMB", "python3 optimbench.py --optimizer FusedAdam --fp16", "python3 optimbench.py --optimizer FusedAdam"] + \
    ["multi-process-adam-scattered lamb", "multi-process-adam-scattered adam", "multi-process-adam-scatteredf16 adam", "multi-process-adam-scatteredf16 lamb"]
all_gpus = [2**i for i in range(1, 9)] 
all_channels = [2,16,32,64,80]
all_algos = ["ring", "tree"]
all_protocols = ["ll", "ll128", "simple", "default"]
all_sizes = [2**i for i in range(10, 30+1)] + [335708160]
bert_layer_size = 335708160

def slurp(file_path):
    f = open(file_path, "r")
    s = f.read()
    f.close()

    return s

#Get data from the job's name
def binary_from_job_name(job_name):
    return re.findall(r'binary=(.+)-p1', job_name)[0]
def gpu_from_job_name(job_name):
    return re.findall(r'gpu=(.+?)-', job_name)[0]
def channels_from_job_name(job_name):
    return re.findall(r'channels=(.+?)-', job_name)[0]
def algo_from_job_name(job_name):
    return re.findall(r'algo=(.+?)-', job_name)[0]
def protocol_from_job_name(job_name):
    return re.findall(r'protocol=(.+?)!', job_name)[0]

#Process stdout from each binary
def process_stdout(stdout_txt):
    all_data = re.findall(r"{.+}", stdout_txt)
    data_in_dict = {}
    for i in all_data:
        i = i.replace("{", '{"')
        i = i.replace(":", '":')
        i = i.replace(",", ',"')
        j = ast.literal_eval(i)
        for k in dict(j):
            j[k.strip()] = j[k]
            if (k != k.strip()):
                j.pop(k)

        data_in_dict[j["SZ"]] = j
    return data_in_dict

# A Dictionary of Binary X # of GPUs X # of Channel X Algorithms X Protocols
full_data_dict = {}

for binary in all_binaries:
    full_data_dict[binary] = {}
    for gpu in all_gpus:
        full_data_dict[binary][gpu] = {}
        for channel in all_channels:
            full_data_dict[binary][gpu][channel] = {}
            for algo in all_algos:
                full_data_dict[binary][gpu][channel][algo] = {}
                for protocol in all_protocols:
                    full_data_dict[binary][gpu][channel][algo][protocol] = {}

def process_dir(_dir):
    f = os.path.join(_dir, "json.json")
    command = slurp(f)
    binary = ""
    for b in all_binaries:
        if b in command:
            binary = b
            break

    gpus = int(re.findall(r"-np (\d+)", command)[0])
    channels = int(re.findall(r"NCCL_MIN_NCHANNELS=(\d+)", command)[0])
    algo = re.findall(r"NCCL_ALGO=(\w+)", command)[0].lower()
    if "NCCL_PROTO" in command:
        protocol = re.findall(r"NCCL_PROTO=([\w\d]+)", command)[0].lower()
    else:
        protocol = "default"

    assert binary in all_binaries, "Possible invalid binary name '%s'"%binary
    assert gpus in all_gpus, "Possible invalid number of gpus '%s'"%gpus
    assert channels in all_channels, "Possible invalid number of channels '%s'"%channels
    assert algo in all_algos, "Possible invalid number of algo '%s'"%algo
    assert protocol in all_protocols, "Possible invalid number of protocol '%s'"%protocol
    stdout_txt = slurp(os.path.join(_dir, "stdout.txt"))
    data = process_stdout(stdout_txt)
    global full_data_dict
    prev_data = full_data_dict[binary][gpus][channels][algo][protocol]
    if (len(data) == 0):
        return    
    full_data_dict[binary][gpus][channels][algo][protocol] = data
    
def get_time(d):
    if "TotalTime" in d:
        return d["TotalTime"]
    if "Total" in d:
        return d["Total"]
    if "Time" in d:
        return d["Time"]
    raise Exception("Time not found in " + str(d))

def process_results_dir(results_dir):
    for d in os.listdir(results_dir):
        full_path = os.path.join(results_dir, d)
        if os.path.isdir(full_path):
            process_dir(full_path)
