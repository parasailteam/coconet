import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys 

csv_file = sys.argv[1] #
optimizer = 'Adam' if 'adam' in csv_file else 'LAMB'

x = []
y = []
with open(csv_file, 'r') as f:
    reader = csv.reader(f, delimiter=' ', quotechar='|')
    for index, row in enumerate(reader):
        
        if index == 0:
            x = row[1:-1]
        else:
            y = row[1:]

optims = []
single_buffer_times = []
scattered_buffer_times = []

for i in range(len(x)):
    if 'scattered' not in x[i]:
        optim = x[i].replace("test-","").capitalize()
        if '16' not in optim:
            optim = optim + 'FP32'
        else:
            optim = optim.replace('f16', '')+'MP'

        optims += [optim]
        single_buffer_times += [float(y[i])]
    else:
        scattered_buffer_times += [float(y[i])]

x_pos = np.arange(len(optims))
width = 0.35

plt.bar(x_pos, single_buffer_times, width, color='green', label='Single Buffer')
plt.bar(x_pos+width, scattered_buffer_times, width, color='red', label='Scattered Buffer')
plt.legend(loc='best')
plt.ylabel('Time in milliseconds', fontsize=14)
plt.xticks(x_pos+width/2, optims)

FIGURES_DIR = "../../../figures/"
fig = plt.gcf()
fig.savefig(FIGURES_DIR+csv_file.replace(".csv", ".pdf"))

