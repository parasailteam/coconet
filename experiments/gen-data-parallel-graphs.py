import subprocess
import sys
import os
import math
import random
import datetime
import shutil

FULL_PERF_EVAL = "1000"
epochs = 1

if len(sys.argv) < 2:
    print("Results directory not specified")
    sys.exit(0)

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

# Generate Graphs for Adam
try:
    print("Generating graphs")
    currDir = os.getcwd()
    os.chdir("./generate-pdf")
    c = "python3 dsl_vs_apex.py --optimizer adam --ranks %s --channels 32 --protocol simple --results-dir %s"%(nranks, resultsDir)
    results_csv = "results-adam-%s.csv"%(nranks)
    results_pdf = results_csv.replace(".csv", ".pdf")
    s,o = execute_command(c)
    with open(results_csv, "w") as f:
        f.write(o)
    c = "python3 plot_dsl_vs_apex.py results-adam-%s.csv"%(nranks)
    execute_command(c)
    os.chdir(currDir)
    os.rename(results_pdf, "Figure10a.pdf")
except Exception as e:
    print(e)

# Generate Graphs for LAMB
try:
    print("Generating graphs")
    currDir = os.getcwd()
    os.chdir("./generate-pdf")
    c = "python3 dsl_vs_apex.py --optimizer lamb --ranks %s --channels 32 --protocol simple --results-dir %s"%(nranks, resultsDir)
    results_csv = "results-lamb-%s.csv"%(nranks)
    results_pdf = results_csv.replace(".csv", ".pdf")
    s,o = execute_command(c)
    with open(results_csv, "w") as f:
        f.write(o)
    c = "python3 plot_dsl_vs_apex.py results-lamb-%s.csv"%(nranks)
    execute_command(c)
    os.chdir(currDir)
    os.rename(results_pdf, "Figure10b.pdf")
except Exception as e:
    print(e)

# try:
#     #Perf eval LAMB FP32
#     os.chdir(nccl_path)
#     print("In parent dir: %s"%nccl_path)
#     checkout_branch("accc-dsl-lamb-moments-distributed")
#     compile_nccl(os.path.join(nccl_path, "nccl-2/"))
#     os.chdir(os.path.join(nccl_path, "accc-dsl/example/"))
#     eval_binary("./allreduce-lamb", epochs, ldLibraryPath)
#     eval_binary("./reducescatter-lamb-allgather", epochs, ldLibraryPath)
#     eval_binary("./test-lamb", epochs, ldLibraryPath)
# except Exception as e:
#     print (e)
    

# try:
#     #Perf eval LAMB FP16
#     os.chdir(nccl_path)
#     print("In parent dir: %s"%nccl_path)
#     checkout_branch("mixed-precision-allgather-fp16weights-LAMB")
#     compile_nccl(os.path.join(nccl_path, "nccl-2/"))
#     os.chdir(os.path.join(nccl_path, "accc-dsl/example/"))
#     # eval_binary("./allreduce-lambf16", epochs, ldLibraryPath)
#     # eval_binary("./reducescatter-lamb-allgatherf16", epochs, ldLibraryPath)
#     eval_binary("./test-lambf16", epochs, ldLibraryPath)
# except Exception as e:
#     print (e)
    
#Switch back to original branch
os.chdir(nccl_path)
