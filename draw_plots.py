import matplotlib.pyplot as plt
import numpy as np

file1 = open('xl_net_results.txt', 'r')
Lines = file1.readlines()
xticks = [1,5,10,15,20,25,30,35]

fig, ax = plt.subplots()
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.rainbow(np.linspace(0, 1, 23))))
count = 0
# Strips the newline character
for line in Lines:
    count +=1
    if count % 2 == 0 and count!=2:
        line = line.strip().strip('][').split(', ')
        results = [float(i) for i in line]
        print(results)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Accuracy')
        ax.plot(xticks, results, label='layer' + str(int(count / 2)))
        ax.legend()
        ax.set_xlim(0, 45)
plt.title('XLNet prediction accuracy on fMRI data')
plt.show()


file1 = open('results.txt', 'r')
Lines = file1.readlines()


fig, ax = plt.subplots()
plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.rainbow(np.linspace(0, 1, 12))))

count = 0
# Strips the newline character
for line in Lines:
    count +=1
    if count % 2 == 0 and count !=16:
        line = line.strip().strip('][').split(', ')
        results = [float(i) for i in line]
        print(results)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Accuracy')
        ax.plot(xticks, results, label= 'layer' + str(int(count/2)))
        ax.legend()
        ax.set_xlim(0, 45)

plt.title('BERT prediction accuracy on fMRI data')
plt.show()