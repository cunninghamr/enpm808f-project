import numpy as np
import os
import time


class Metrics:
    def __init__(self, name=None):
        if name is None:
            self.name = time.strftime('%Y%m%d%H%M%S')
        else:
            self.name = name

        self.num_episodes = 0
        self.episode_reward = []
        self.episode_avg_max_q = []

    def save(self, num):
        if not os.path.isdir('results'):
            os.mkdir('results')
        if not os.path.isdir(os.path.join('results', self.name)):
            os.mkdir(os.path.join('results', self.name))
        if not os.path.isdir(os.path.join('results', self.name, str(num))):
            os.mkdir(os.path.join('results', self.name, str(num)))
        np.savetxt(os.path.join('results', self.name, 'reward_{}.txt'.format(num)), np.asarray(self.episode_reward), fmt='%.4f',
                   delimiter=',')
        np.savetxt(os.path.join('results', self.name, 'avg_max_q_{}.txt'.format(num)), np.asarray(self.episode_avg_max_q), fmt='%.4f',
                   delimiter=',')
