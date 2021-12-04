import argparse
from process_hypercluster_data import *

parser = argparse.ArgumentParser(description='Generate times of DSL generated Adam/Lamb vs FusedAdam/FusedLamb')
parser.add_argument('--optimizer', type=str, required=True)
parser.add_argument('--fp16',action='store_true',default=False)
parser.add_argument('--ranks', type=int,required=True)
parser.add_argument('--channels', type=str,required=True)
parser.add_argument('--protocol', type=str,required=False,default="minimum")
parser.add_argument('--scatteredptrs', action='store_true', default=False, required=False)

args = parser.parse_args()

optimizer = args.optimizer.lower()

assert optimizer in ['adam', 'lamb', 'all']
MIN_OF_ALL_CHANNELS = "minimum"
ranks = args.ranks 
ranks in all_gpus
channels = args.channels
if channels != MIN_OF_ALL_CHANNELS:
    channels in all_channels

MIN_OF_ALL_PROTOCOLS = "minimum"
assert args.protocol  == MIN_OF_ALL_PROTOCOLS or args.protocol in all_protocols

filtered_binaries = []
apex_binaries = []
filtered_sizes = []
if not args.scatteredptrs:
    filtered_sizes = all_sizes
    filtered_sizes.remove(bert_layer_size)
else:
    filtered_sizes = [bert_layer_size]

if (not 'all' in optimizer) != (not args.scatteredptrs):
    raise Exception("all optimizers are allowed only for scattered ptrs")

for binary in all_binaries:
    if args.scatteredptrs:
        include_binary = ("scattered" not in binary and optimizer in binary.lower())
        include_binary = include_binary or ("scattered" in binary and optimizer in  binary[binary.find("scattered"):])
        include_binary = include_binary or (optimizer == 'all')
        include_binary = include_binary and ("scattered" in binary or 'test-' in binary)

        if include_binary:
            include_binary = args.fp16 and '16' in binary
            include_binary = include_binary or (not args.fp16 and '16' not in binary)
            include_binary = include_binary or optimizer == 'all'
            if include_binary:
                filtered_binaries += [binary]
    else:
        if 'scattered' in binary:
            continue
        if optimizer in binary.lower():
            if args.fp16:
                if 'f16' in binary.lower() or '--fp16' in binary.lower():
                    if 'python' in binary:
                        apex_binaries += [binary]
                    else:
                        filtered_binaries += [binary]
            elif 'f16' not in binary.lower() and '--fp16' not in binary.lower():
                if 'python' in binary:
                    apex_binaries += [binary]
                else:
                    filtered_binaries += [binary]

if not args.scatteredptrs:
    assert len(apex_binaries) == 1, "Apex Baselines are not 1: " + str(apex_binaries)
else:
    assert len(apex_binaries) == 0, "There should be no apex baselines in scattered-ptrs case but there are: " + str(apex_binaries)
if args.scatteredptrs and optimizer == 'all':
    #Combine binaries of same optimizer with each other
    filtered_binaries = ['test-adam','multi-process-adam-scattered adam', 'test-lamb', 'multi-process-adam-scattered lamb', 'test-adamf16', 'multi-process-adam-scatteredf16 adam', 'test-lambf16', 'multi-process-adam-scatteredf16 lamb']

firstRow = ["Size"] + filtered_binaries + ["Fused%s"%optimizer.capitalize()] + ["AllReduceBW"] #[x+"-BW" for x in filtered_binaries] + ["Fused%s-BW"%optimizer.capitalize()]
formatString = "{:<15} " * len(firstRow)
rows_to_print = [firstRow]

def binary_contains_allreduce_time(binary):
    # return  binary == "adam" or binary == "allreduce-lamb" or binary == "allreduce-adamf16" or binary == "allreduce-lambf16"
    return "ar-c" in binary

def get_apex_c_time(times):
    if optimizer == 'adam':
        return times["Total"]# - times["FusedAdam"] + times["FusedAdam-CTime"]
    elif optimizer == 'lamb':
        return times["Total"]# - times["FusedLAMB"] + times["FusedLAMB-CTime"]

def bandwidth(t, size, ranks, epochs):
    b = 2 * (ranks-1)/ranks * size*(2 if args.fp16 else 4) / (t/epochs * 1e-3) 
    b_in_gbps = b/(2**30)
    return b_in_gbps

def get_protocol_for_size(size):
    assert(size > 0)
    if size < 1*1024*1024:
        return "ll"
    if size < 16*1024*1024:
        return "simple"
    return "simple"

for size in filtered_sizes:
    row = [size]
    bandwidth_row = []
    allreduce_time = 1
    if not args.scatteredptrs:
        apex_time = 1 << 30
        for channel in ([int(channels)] if channels != MIN_OF_ALL_CHANNELS else all_channels):
            protocol_times = full_data_dict[apex_binaries[0]][ranks][channel]["ring"]
            for protocol in protocol_times:
                if args.protocol != MIN_OF_ALL_PROTOCOLS and protocol != args.protocol:
                    continue
                if args.protocol == MIN_OF_ALL_PROTOCOLS and protocol != get_protocol_for_size(size):
                    continue
                if size in protocol_times[protocol]:
                    # if binary == "test-adam":
                    if (get_apex_c_time(protocol_times[protocol][size]) < apex_time):
                        apex_time = min(get_apex_c_time(protocol_times[protocol][size]), apex_time)
                        # print(size, protocol, apex_time)
                        #allreduce_time = protocol_times[protocol][size]["AllReduce"]
                # else:
                #     if binary == "test-adam":
                #         print ("no result for ",size, "for ", binary, " for ", protocol)
        if apex_time == 1<<30 and not args.scatteredptrs:
            assert "Cannot find apex_time"
    epochs = -1
    allreduce_adam_time = 0
    for binary in filtered_binaries:
        min_time = 1 << 30
        for channel in ([int(channels)] if channels != MIN_OF_ALL_CHANNELS else all_channels):
            protocol_times = full_data_dict[binary][ranks][channel]["ring"]
            for protocol in protocol_times:
                if args.protocol != MIN_OF_ALL_PROTOCOLS and protocol != args.protocol:
                    continue
                if args.protocol == MIN_OF_ALL_PROTOCOLS and protocol != get_protocol_for_size(size):
                    continue

                if size in protocol_times[protocol]:
                    epochs = protocol_times[protocol][size]["Epochs"]
                    if binary_contains_allreduce_time(binary):
                        if get_time(protocol_times[protocol][size]) < min_time:
                            min_time = get_time(protocol_times[protocol][size])
                            allreduce_time = protocol_times[protocol][size]["AllReduce"]
                    else:
                        min_time = min(get_time(protocol_times[protocol][size]), min_time)
        if binary_contains_allreduce_time(binary):
            allreduce_adam_time = min_time
        # print(size,binary,protocol, allreduce_adam_time)
        row += [min_time]
        # if (binary == "test-adamf16"):
        #     print(size, min_time, apex_time)
        # min_time = bandwidth(min_time, size, ranks, epochs)
        bandwidth_row += [min_time]
    # allreduce_time = bandwidth(allreduce_time, size, ranks)
    max_improvement = min(row[1:]+[allreduce_time])
    # print (size, row)
    for i in range(1, len(row)):
        row[i] = row[i] if args.scatteredptrs else apex_time/row[i]# max_improvement/row[i]

    # bandwidth_row += [bandwidth(apex_time, size, ranks, epochs), bandwidth(allreduce_adam_time, size, ranks, epochs)]
    bandwidth_row = [bandwidth(allreduce_time, size, ranks, epochs)]
    rows_to_print += [row + [apex_time/apex_time] + [apex_time/max_improvement*1.02]]#[row+([max_improvement/apex_time] if not args.scatteredptrs else []) + bandwidth_row]

csvwriter = csv.writer(sys.stdout, delimiter=' ',
                       quotechar='|', quoting=csv.QUOTE_MINIMAL)
csvwriter.writerows(rows_to_print)
sys.stdout.flush()