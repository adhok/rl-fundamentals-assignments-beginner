# rl/environment/bandits/k_armed_bandit.py

import numpy as np

class KArmedTestbed:
    def __init__(self, num_runs, k, k_mean, k_std, bandit_std, seed):
        self.num_runs = num_runs
        self.k = k
        self.k_mean = k_mean
        self.k_std = k_std
        self.bandit_std = bandit_std
        np.random.seed(seed)
        self.bandits = [Bandit(k, k_mean, k_std, bandit_std) for _ in range(num_runs)]

class Bandit:
    def __init__(self, k, k_mean, k_std, bandit_std):
        self.k = k
        self.bandit_std = bandit_std
        self.q_star = np.random.normal(k_mean, k_std, k)
        self.best_action = np.argmax(self.q_star)

    def step(self, action):
        return np.random.normal(self.q_star[action], self.bandit_std)