import subprocess
import sys
import os
import math
import random
import datetime

FULL_PERF_EVAL = "1000"
epochs = ""

nccl_path = ""
print (""" This script must be copied to the parent directory of nccl directory. 
Takes 5 command line args:
    <path to nccl repository> <path to results directory> <start # of ranks> <end # of ranks> optional: <epochs>
""")

assert len(sys.argv) >= 4, "Provide <path to nccl repository> <start # of ranks> <end # of ranks>"

nccl_path = sys.argv[1]
resultsDir = sys.argv[2]
startRanks = float(sys.argv[3])
endRanks = float(sys.argv[4])

assert startRanks == 1 << int(math.log(startRanks, 2)), "# of start ranks must be power of two"
assert endRanks == 1 << int(math.log(endRanks, 2)), "# of end ranks must be power of two"

if len(sys.argv) >= 6:
    epochs = sys.argv[5]

resultsDir = os.path.abspath(resultsDir)
if not os.path.exists(resultsDir):
    print("Making ", resultsDir)
    os.mkdir(resultsDir)
else:
    print("results dir ", resultsDir, " already exists. Create a new dir.")
    sys.exit(0)

#Run nvprof with mpi on single machine
#sudo PYTHON_PATH="/home/parasail/.pyenv/versions/myvenv/lib/:$PYTHON_PATH" /usr/local/cuda/bin/nvprof  --profile-child-processes mpirun -np 4 -x NCCL_ALGO=Ring -x MASTER_ADDR=127.0.0.1 -x MASTER_PORT=10000 --allow-run-as-root /home/parasail/.pyenv/shims/python optimbench.p

#Value of epochs other than FULL_PERF_EVAL means for each size exactly 'epochs' number of iterations
#will be executed otherwise for different sizes different number of 'epochs' will be executed.


parent_dir = os.getcwd()
nccl_path = os.path.join(parent_dir, nccl_path)
#Check that current directory is not same as nccl_path
print ("Current dir", parent_dir, "NCCL path", nccl_path)
assert os.path.normpath(parent_dir) != os.path.normpath(nccl_path), "perf-eval.py is in same path as nccl's path. This is not allowed. Please place this file in the parent directory."
assert os.path.exists(nccl_path), "Invalid nccl path"

#Modify the address and port
# if ("PHILLY_CONTAINER_IP" not in os.environ):
    # addressPort = "-x MASTER_ADDR=127.0.0.1 -x MASTER_PORT=10000"
# else:
addressPort = " -hostfile /job/hostfile -x MASTER_ADDR=worker-0 -x MASTER_PORT=10000 "

def pull_all_branches():
    os.chdir(nccl_path)
    s, o = subprocess.getstatusoutput("git pull --all")
    if (s != 0):
        raise Exception("git pull unsuccessfull in '%s'\n"%(nccl_path) + 
                        "git message: \n%s"%o)
    else:
        print("Pull successfull")
        
def checkout_branch(branch):
    os.chdir(nccl_path)
    s, o = subprocess.getstatusoutput("git checkout %s"%branch)
    if (s != 0):
        raise Exception("Cannot checkout branch '%s' in '%s'\n"%(branch, nccl_path) + 
                        "git message: \n%s"%o)
    else:
        print("Checkout of '%s' successfull"%branch)

def compile_nccl(path):
    print("Compiling nccl at '%s'"%path)
    assert os.path.exists(path), "Cannot commpile nccl because of invalid path"
    os.chdir(path)
    s, o = subprocess.getstatusoutput("rm -rf build/ ; make -j src.build NVCC_GENCODE=\"-gencode=arch=compute_70,code=sm_70\"")
    if (s != 0):
        raise Exception("nccl compilation unsuccessful in '%s'\n"%(path) + 
                        "make output: \n%s"%o)
    else:
        print("nccl compiled succesfully")

def make_binary(binary):
    if "python" in binary:
        return
    print("make " + binary)
    s, o = subprocess.getstatusoutput("rm " + binary + " ; " + " make "+ binary)
    if s != 0:
        raise Exception("make unsuccesfull.\n" +o )
    else:
        print("make succesfull")

def new_application_dir():
    while True:
        d = datetime.datetime.now().date()
        t = datetime.datetime.now().time()
        dirid = "{}_{}_{}_{}_{}_{}".format(d.day, d.month, d.year, t.hour, t.minute, t.second)
        appDir = os.path.join(resultsDir, "application_"+str(dirid))
        if (os.path.exists(appDir)):
            continue

        os.mkdir(appDir)
        return appDir

nchannels = [32,64]
def eval_binary(binary, epochs, envVars=""):
    make_binary(binary)
    if (startRanks == endRanks):
        nranks = [int(startRanks)]
    else:
        nranks = [int(2**i) for i in range(int(math.log(startRanks, 2)), int(math.log(endRanks, 2))+1)]
    print ("Running on ", nranks, " ranks ")
    for algo in ["Ring"]:
        for proto in ["Simple", "LL", "LL128"]:
            for channels in nchannels: #Channels
                if "allreduce-" in binary or "reducescatter" in binary or "python" in binary:
                    if channels == 64:
                        continue
                for ranks in nranks: #Ranks
                    for nthreads in [640]:#range(192, 640, 32):
                        appDir = new_application_dir()
                        storeOutFile = os.path.join(appDir, "stdout.txt")
                        p = "-x NCCL_ALGO=" + algo + " -x NCCL_PROTO=" + proto
                        #c = "mpirun -np 16 " + p + " -x LD_LIBRARY_PATH=../../nccl-2/build/lib/ -x MASTER_ADDR=$PHILLY_CONTAINER_IP -x MASTER_PORT=$((PHILLY_CONTAINER_PORT_RANGE_START+25)) -hostfile ~/hostfile python optimbench.py"
                        _envVars = envVars + "  -x NCCL_MIN_NCHANNELS=%d -x NCCL_NTHREADS=%d -x NCCL_LL128_NTHREADS=%d -x NCCL_MAX_NCHANNELS=%d -x NCCL_BUFFSIZE=4194304 "%(channels, nthreads, nthreads, channels)
                        c = "mpirun -mca btl_tcp_if_include enp134s0f1 -np " + str(ranks) +" " + p + " " + _envVars + addressPort + " "+binary + " " + (epochs if "python" not in binary else ("--times " + epochs if epochs != "" else ""))
                        c = c + " &> " + storeOutFile
                        print (c)
                        print("starting at ", str(datetime.datetime.now()))
                        s, o = subprocess.getstatusoutput(c)
                        
                        print("done at ", str(datetime.datetime.now()))
                        print("Storing results in  ", appDir)
                        with open(os.path.join(appDir, "json.json"), "w") as f:
                            f.write(c)
                        # with open(os.path.join(appDir, "stdout.txt"), "w") as f:
                        #     f.write(o)
                        # with open(os.path.join(parent_dir, 'results-gpus.txt'), 'a') as f:
                        #     #f.write(o[o.find("<results>") + len("<results>"): o.find("</results>")])
                        #     f.write(c + "\n" + o+"</results>\n")
                        # if (s == 0):
                        #     print "Executed Successfully."
                        # else:
                        #     print "Error in execution. Output: ", o

print("pull all branches")
os.chdir(nccl_path)
# pull_all_branches()

ldLibraryPath = "-x LD_LIBRARY_PATH=../../nccl-2/build/lib/:$LD_LIBRARY_PATH"
pythonpath = ' -x PYTHONPATH=\\"../../:$PYTHONPATH\\" '
#Perf eval FusedAdam FP32
# try:
#     os.chdir(nccl_path)
#     print("In parent dir: %s"%nccl_path)    
#     checkout_branch("accc-dsl-moments-distributed")
#     os.chdir(os.path.join(nccl_path, "accc-dsl/example/"))    
#     eval_binary("python3 optimbench.py --optimizer FusedAdam", epochs, pythonpath)
#     eval_binary("python3 optimbench.py --optimizer FusedAdam --fp16", epochs, pythonpath)
#     eval_binary("python3 optimbench.py --optimizer FusedLAMB", epochs, pythonpath)
#     eval_binary("python3 optimbench.py --optimizer FusedAdam --fp16", epochs, pythonpath)
# except Exception as e:
#     print (e)

# sys.exit(0)

#Perf eval Adam FP32
# try:
if False:
    os.chdir(nccl_path)
    print("In parent dir: %s"%nccl_path)    
    checkout_branch("accc-dsl-moments-distributed")
    compile_nccl(os.path.join(nccl_path, "nccl-2/"))
    os.chdir(os.path.join(nccl_path, "accc-dsl/example/"))
    eval_binary("./allreduce-adam", epochs, ldLibraryPath)
    eval_binary("./reducescatter-adam-allgather", epochs, ldLibraryPath)
    eval_binary("./test-adam", epochs, ldLibraryPath)
# except Exception as e:
#     print (e)
    
#Perf eval Adam FP16
# try:
#     os.chdir(nccl_path)
#     print("In parent dir: %s"%nccl_path)
#     checkout_branch("mixed-precision-allgather-fp16weights")
#     compile_nccl(os.path.join(nccl_path, "nccl-2/"))
#     os.chdir(os.path.join(nccl_path, "accc-dsl/example/"))
#     eval_binary("./allreduce-adamf16", epochs,ldLibraryPath)
#     eval_binary("./reducescatter-adam-allgatherf16", epochs,ldLibraryPath)
#     eval_binary("./test-adamf16", epochs,ldLibraryPath)
# except Exception as e:
#     print (e)

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
    

try:
    #Perf eval LAMB FP16
    os.chdir(nccl_path)
    print("In parent dir: %s"%nccl_path)
    checkout_branch("mixed-precision-allgather-fp16weights-LAMB")
    compile_nccl(os.path.join(nccl_path, "nccl-2/"))
    os.chdir(os.path.join(nccl_path, "accc-dsl/example/"))
    # eval_binary("./allreduce-lambf16", epochs, ldLibraryPath)
    # eval_binary("./reducescatter-lamb-allgatherf16", epochs, ldLibraryPath)
    eval_binary("./test-lambf16", epochs, ldLibraryPath)
except Exception as e:
    print (e)
    
#Switch back to original branch
print("Switching back to accc-dsl-moments-distributed")
os.chdir(nccl_path)
print("In parent dir: %s"%nccl_path)
checkout_branch("accc-dsl-moments-distributed")

# eval_binary("./reducescatter-adam-allgather")
# eval_binary("./test-adam")
# eval_binary("python3 optimbench.py")
