import re

f = open("adam-results-128-gpus-all-algos", "r")
ourAdam = f.read()
f.close()

f = open("/philly/rr3/msrhyperprojvc2_scratch/saemal/abhinav/nccl-manual/samples/optim-bench-results-128GPUs", "r")
otherAdams = f.read()
f.close()

adamResults = {"FusedAdam":{}, "PyTorchAdam":{}, "OurAdam":{}} #dictionary of [FusedAdam, PyTorchAdam, OurAdam]x[Sizes]x[Times]
allSizes = []
for size, time in re.findall(r'\(null\) (\d+) ([\d\.]+)', ourAdam):
    adamResults["OurAdam"][int(size)] = float(time)
    allSizes += [int(size)]

for size, time in re.findall(r'fusedadam (\d+) \d+ ([\d\.]+)', otherAdams):
    adamResults["FusedAdam"][int(size)] = float(time)

for size, time in re.findall(r'adam (\d+) \d+ ([\d\.]+)', otherAdams):
    adamResults["PyTorchAdam"][int(size)] = float(time)

print ("{:<15} {:<15} {:<15} {:<15} {:<15} {:<15}".format("Size", "FusedAdam", "PyTorchAdam", "OurAdam", "Speedup Over FusedAdam", "Speedup Over PytorchAdam")) 

for sz in allSizes:
    print("{:<15} {:<15.2f} {:<15.2f} {:<15.2f} {:<20.2f} {:<20.2f}".format(sz, adamResults["FusedAdam"][sz], adamResults["PyTorchAdam"][sz], adamResults["OurAdam"][sz], 
          adamResults["FusedAdam"][sz]/adamResults["OurAdam"][sz],  adamResults["PyTorchAdam"][sz]/adamResults["OurAdam"][sz]))
