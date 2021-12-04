import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys 

csv_file = sys.argv[1] #
optimizer = 'Adam' if 'adam' in csv_file else 'LAMB'

x = []
y = [[],[],[],[]] #relative bandwidth
bwy = [[],[],[],[]] #absolute bandwidth numbers
pivot_point_x = 0
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    for index, row in enumerate(reader):
        print(f"[DEBUG] index = {index}, row = {row}")
        if index > 0: # skip header and 512 size
            data = [float(i) for i in row[0].split()]
            if (data[5] > 1e6):
                continue
            x.append(int(data[0]))
            ar_fa = data[4]
            y[0].append(data[1])
            y[1].append(data[2])
            y[2].append(data[3])
            y[3].append(data[5])
            bwy[0].append(data[5])
            if (float(data[1]) > float(data[3])):
                pivot_point_x+=1
            # bwy[1].append(data[6])
            # bwy[2].append(data[7])
            # bwy[3].append(data[8])

# pivot_point_x += 1
fig, ax = plt.subplots()
#ax.plot(x, y)
# if optimizer == "Adam" and "fp16" in csv_file:
#     y[1][len(y[2]) - 1] 
#     print (y[3], y[3][17:], len(y[3]))
op = optimizer[0]
print(pivot_point_x)
# ax.semilogx(x,y[0],'o-',x,y[1],'s-',x,y[2],'^-',x,y[3],'D-',base=2)
l = ax.semilogx(x,y[3],'D-',x[:pivot_point_x],y[0][:pivot_point_x],'o-',x[pivot_point_x:],y[2][pivot_point_x:],'^-',x,y[1],'s-',base=2)
ax.semilogx(x[pivot_point_x-1:],y[0][pivot_point_x-1:],'o--',color=l[1].get_color(),base=2)
ax.semilogx(x[:pivot_point_x+1],y[2][:pivot_point_x+1],'^--',color=l[2].get_color(),base=2)
# ax.semilogx(x[:7],y[0][:7],'o-',x,y[1],'s-',x[6:],y[2][6:],'^-',x,y[3],'D-',base=2)
plt.xlabel('# of Elements in Tensor', fontsize=14)
plt.ylabel('Speedup over\n AllReduce+Fused%s'%optimizer, fontsize=14, ha="center")
plt.setp(ax.get_xticklabels(), fontsize=14)
plt.setp(ax.get_yticklabels(), fontsize=14)
plt.xticks([2**x for x in range(10, 31, 2)])
# plt.axis('off')
# if optimizer == "Adam":
#     ax.set_ylim([0.6,1.9])
# else:
#     ax.set_ylim([0.6,1.9])
ax.grid()



# ax.get_yaxis().set_ticklabels([])

# legend = ax.legend(("AR-"+optimizer,"RS-%s-AG"%(optimizer),"fuse(RS-%s-AG)"%optimizer, "Fused%s"%optimizer), loc='upper left', shadow=True, fontsize='large',bbox_to_anchor=(-0.01, 1.2),ncol=3)

# legend = ax.legend(("T1","T2","T3", "Fused%s"%optimizer), loc='upper left', shadow=True, fontsize='large',bbox_to_anchor=(-0.01, 1.12),ncol=4)

# legend = plt.legend(("CoCoNet(AR-%s)"%op,"RS-%s-AG"%op,"CoCoNet(fuse(RS-%s-AG))"%op, "AR"), loc='upper left', fontsize='large',bbox_to_anchor=(-0.03, 1.2),ncol=4,columnspacing=1,handlelength=1.7)
legend = plt.legend(("UB", "CoCoNet(AR-%s)"%op,"CoCoNet(fuse(RS-%s-AG))"%op,"GShard-Eq"), loc='upper left', fontsize=11.4,bbox_to_anchor=(-0.18, 1.2),ncol=4,columnspacing=0.5,handlelength=1.5)
# axBW = ax.twinx()
# axBW.plot(x,bwy[0],'*-')
# axBW.set_ylabel('Bandwidth (GB/s)', fontsize=14)
# plt.legend(("AR-BW"))
#plt.tight_layout()

FIGURES_DIR = "../"
# FIGURES_DIR = "./"
fig = plt.gcf()
fig.set_size_inches(6.4,3.5)

fig.savefig(FIGURES_DIR+csv_file.replace(".csv", ".pdf"),pad_inches=0,bbox_inches='tight')
print(fig.get_size_inches())
# plt.show()
