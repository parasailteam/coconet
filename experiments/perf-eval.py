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
    <path to nccl repository> <path to results directory>
""")

assert len(sys.argv) >= 4, "Provide <path to nccl repository> <start # of ranks> <end # of ranks>"
nranks = os.environ.get("NPROC")
nccl_path = sys.argv[1]
resultsDir = sys.argv[2]

resultsDir = os.path.abspath(resultsDir)
if not os.path.exists(resultsDir):
    print("Making ", resultsDir)
    os.mkdir(resultsDir)
else:
    print("results dir ", resultsDir, " already exists. Create a new dir.")
    sys.exit(0)

#Run nvprof with mpi on single machine
#sudo PYTHON_PATH="/home/parasail/.pyenv/versions/myvenv/lib/:$PYTHON_PATH" /usr/local/cuda/bin/nvprof  --profile-child-processes mpirun -np 4 -x NCCL_ALGO=Ring -x MASTER_ADDR=127.0.0.1 -x MASTER_PORT=10000 --allow-run-as-root /home/parasail/.pyenv/shims/python optimbench.p

parent_dir = os.getcwd()
nccl_path = os.path.join(parent_dir, nccl_path)
#Check that current directory is not same as nccl_path
print ("Current dir", parent_dir, "NCCL path", nccl_path)
assert os.path.exists(nccl_path), "Path to nccl '%s' is invalid"%nccl_path


mpiargs = "-mca btl_tcp_if_include enp134s0f1"
if "MPI_ARGS" in os.environ:
    mpiargs += os.environ.get("MPI_ARGS") #" -hostfile /job/hostfile -x MASTER_ADDR=worker-0 -x MASTER_PORT=10000 "

def compile_nccl(path):
    print("Compiling nccl at '%s'"%path)
    assert os.path.exists(path), "Path to nccl '%s' is invalid"%path
    os.chdir(path)
    s, o = subprocess.getstatusoutput("rm -rf build/ ; make -j src.build NVCC_GENCODE=\"-gencode=arch=compute_70,code=sm_70\"")
    if (s != 0):
        raise Exception("nccl compilation unsuccessful in '%s'\n"%(path) + "make output: \n%s"%o)
    else:
        print("nccl compiled succesfully")

def make_binary(binary):
    if "python" in binary:
        return
    print("make " + binary)
    s, o = subprocess.getstatusoutput("rm " + binary + " ; " + " make "+ binary)
    if s != 0:
        raise Exception("make unsuccessful.\n" +o )
    else:
        print("make successful")

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

nchannels = 32
def eval_binary(binary, epochs):
    make_binary(binary)
    print ("Running on ", nranks, " ranks ")
    algo = "Ring"
    proto = "Simple"
    channels = nchannels
    ranks = nranks
    nthreads = 512
    appDir = new_application_dir()
    ldLibraryPath = "-x LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH"%(os.path.join(nccl_path, "build/lib"))
    # pythonpath = ' -x PYTHONPATH=\\"../../:$PYTHONPATH\\" '

    storeOutFile = os.path.join(appDir, "stdout.txt")
    envVars = "-x NCCL_ALGO=" + algo + " -x NCCL_PROTO=" + proto + \
        " -x NCCL_MIN_NCHANNELS=%d -x NCCL_NTHREADS=%d -x NCCL_LL128_NTHREADS=%d -x NCCL_MAX_NCHANNELS=%d -x NCCL_BUFFSIZE=4194304"%(channels, nthreads, nthreads, channels)   
    command = "mpirun -np " + str(ranks) +" " + envVars + " " + binary #+ " " + (epochs if "python" not in binary else ("--times " + epochs if epochs != "" else ""))
    command += " &> " + storeOutFile
    
    print("starting at ", str(datetime.datetime.now()))
    s, o = subprocess.getstatusoutput(c)
    
    print("done at ", str(datetime.datetime.now()))
    print("Storing results in  ", appDir)
    with open(os.path.join(appDir, "json.json"), "w") as f:
        f.write(c)

#Perf eval FusedAdam
try:
    print("In '%s'"%os.getcwd())
    # eval_binary("python3 optimbench.py --optimizer FusedAdam", epochs, pythonpath)
    eval_binary("python3 optimbench.py --optimizer FusedAdam --fp16", epochs)
    # eval_binary("python3 optimbench.py --optimizer FusedLAMB", epochs, pythonpath)
    eval_binary("python3 optimbench.py --optimizer FusedAdam --fp16", epochs)
except Exception as e:
    print (e)

# sys.exit(0)

#Perf eval Adam FP32
# try:
# if False:
#     os.chdir(nccl_path)
#     print("In parent dir: %s"%nccl_path)    
#     checkout_branch("accc-dsl-moments-distributed")
#     compile_nccl(os.path.join(nccl_path, "nccl-2/"))
#     os.chdir(os.path.join(nccl_path, "accc-dsl/example/"))
#     eval_binary("./allreduce-adam", epochs, ldLibraryPath)
#     eval_binary("./reducescatter-adam-allgather", epochs, ldLibraryPath)
#     eval_binary("./test-adam", epochs, ldLibraryPath)
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
