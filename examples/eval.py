import commands
import sys
opt_type = sys.argv[1]

for algo in ["Default", "Ring", "Tree"]:
    if algo == "Default":
        p = ""
    else:
        p = "-x NCCL_ALGO=" + algo
    c = "mpirun -np 128 " + p + " -x LD_LIBRARY_PATH=../../nccl-2/build/lib/ -x MASTER_ADDR=$PHILLY_CONTAINER_IP -x MASTER_PORT=$((PHILLY_CONTAINER_PORT_RANGE_START+25)) -hostfile ~/hostfile python optimbench.py"
    print ("executing ", c)
    s, o = commands.getstatusoutput(c)
    print(o)
    with open(opt_type+'-results-128-gpus-all-algos', 'a') as f:
        f.write("NCCL_ALGO=" + algo)
        f.write(o[o.find("<results>") + len("<results>"): o.find("</results>")])