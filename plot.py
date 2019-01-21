import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd


data_file = "results/greedy-decoding.txt"

start = 10
with open(data_file, 'r') as din:
	d_lines = din.readlines()

d_lines = [line.strip() for line in d_lines]

d_test = float(d_lines[-1])
# d_weight = float(d_lines[0])

d_lines = d_lines[:-1]

d_train = [float(line) for index, line in enumerate(d_lines) if index%2 == 0][start:]
# d_train_loss, d_train_ppl = zip(*d_train)
d_dev = [float(line) for index, line in enumerate(d_lines) if index%2 != 0][start:]

# epoch = pd.Series(range(0, 100), name="epoch")
labels = ["train-ppl", "dev-ppl"]
# data = np.dstack([v_train, v_dev, c_train, c_dev])
# sns.tsplot(data, time=epoch, condition=labels, value='perplexity')
t = np.arange(100-start) + start
fig, ax = plt.subplots()
l1, = ax.plot(t, d_train, linestyle='solid', label=labels[0])
l2, = ax.plot(t, d_dev, linestyle='solid', label=labels[1])
# l3, = ax.plot(t, d_dev, linestyle='solid', label=labels[2])
# fig.suptitle('Oracle Conditional RNNLM', fontsize=20)
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.legend(handles=[l1, l2])
plt.savefig("plots/" + data_file.split('/')[1].split('.')[0] + ".png")

# print(v_train[-1], v_dev[-1], v_test, c_train[-1], c_dev[-1], c_test)






# d_trains = []
# d_devs = []
# d_lambdas = []
# start =20
# data_file  = "results/unk-cond-seg-rnnlm-3-%d.txt"
# for i in range(1,5):
# 	with open(data_file%i, 'r') as din:
# 		d_lines = din.readlines()


# 	d_lines = [line.strip() for line in d_lines]

# 	d_test = float(d_lines[-1])
# 	d_lambda = float(d_lines[0])
# # d_weight = float(d_lines[0])

# 	d_lines = d_lines[1:-1]

# 	d_train = [float(line.split()[0]) for index, line in enumerate(d_lines) if index%2 == 0][start:]
# # d_train_loss, d_train_ppl = zip(*d_train)
# 	d_dev = [float(line) for index, line in enumerate(d_lines) if index%2 != 0][start:]

# 	d_trains.append(d_train)
# 	d_devs.append(d_dev)
# 	d_lambdas.append(d_lambda)

# # epoch = pd.Series(range(0, 100), name="epoch")
# labels = ["train-ppl", "dev-ppl"]
# # data = np.dstack([v_train, v_dev, c_train, c_dev])
# # sns.tsplot(data, time=epoch, condition=labels, value='perplexity')
# t = np.arange(100-start) + start
# fig = plt.figure()
# fig.subplots_adjust(hspace=0.5)
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)
# ax1.title.set_text('Lambda = 0.2')
# ax2.title.set_text('Lambda = 0.4')
# ax3.title.set_text('Lambda = 0.6')
# ax4.title.set_text('Lambda = 0.8')

# l1, = ax1.plot(t, d_trains[0], linestyle='solid', label=labels[0])
# l2, = ax1.plot(t, d_devs[0], linestyle='solid', label=labels[1])
# ax1.legend()

# l3, = ax2.plot(t, d_trains[1], linestyle='solid', label=labels[0])
# l4, = ax2.plot(t, d_devs[1], linestyle='solid', label=labels[1])
# ax2.legend()

# l5, = ax3.plot(t, d_trains[2], linestyle='solid', label=labels[0])
# l6, = ax3.plot(t, d_devs[2], linestyle='solid', label=labels[1])
# ax3.legend()

# l7, = ax4.plot(t, d_trains[3], linestyle='solid', label=labels[0])
# l8, = ax4.plot(t, d_devs[3], linestyle='solid', label=labels[1])
# ax4.legend()

# l3, = ax.plot(t, d_dev, linestyle='solid', label=labels[2])
# fig.suptitle('Oracle Conditional RNNLM', fontsize=20)
# plt.xlabel('epochs')
# plt.ylabel('perplexity')
# plt.legend(handles=[l1, l2, l3, l4, l5,l6,l7,l8])
# plt.savefig("plots/" + "predicting_bit" + ".png")

# print(v_train[-1], v_dev[-1], v_test, c_train[-1], c_dev[-1], c_test)

