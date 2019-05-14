import matplotlib.pyplot as plt
import numpy as np
import os


def mov_avg(arr):
    mov_avg = []
    for i in range(len(arr)):
        mov_avg.append(
            np.average(arr[max(0, i - int(len(arr) * 0.05)):i + 1]))
    return mov_avg


results_dir = 'demo_60_degrees'

training_batch_nums = 100

ep_avg_max_q = []
ep_reward = []

for i in range(1, training_batch_nums + 1):
    path = os.path.join('results', results_dir, 'avg_max_q_{}.txt'.format(i))
    if os.path.isfile(path):
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                ep_avg_max_q.append(float(line))
                line = f.readline()
    path = os.path.join('results', results_dir, 'reward_{}.txt'.format(i))
    if os.path.isfile(path):
        with open(path, 'r') as f:
            line = f.readline()
            while line:
                ep_reward.append(float(line))
                line = f.readline()

if len(ep_avg_max_q) > 0:
    episodes = range(1, len(ep_avg_max_q) + 1)

    ax = plt.subplot(2, 1, 1)
    plt.title(results_dir)
    plt.ylabel('Average Max Q')
    ax.plot(episodes, ep_avg_max_q, 'b')
    ax.plot(episodes, mov_avg(ep_avg_max_q), 'r')
    ax = plt.subplot(2, 1, 2)
    plt.ylabel('Reward')
    ax.plot(episodes, ep_reward, 'b')
    ax.plot(episodes, mov_avg(ep_reward), 'r')
    plt.xlabel('Episode')

    plt.show(block=True)
