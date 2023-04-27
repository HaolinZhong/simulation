import numpy as np


class UCBArm:
    global_N = 0

    def __init__(self, p):
        self.p = p
        self.p_estimate = 0
        self.N = 0
        self.Ns = 0

    def get_p_estimate(self):
        return self.p_estimate

    def get_outcome(self):
        return np.random.random() < self.p

    def update(self, x):
        UCBArm.global_N += 1
        self.N += 1
        self.Ns += x
        self.p_estimate = self.Ns / self.N

    def sample(self):
        ucb = np.sqrt(2 * np.log(UCBArm.global_N) / self.N) if self.N > 0 else 0
        return self.p_estimate + ucb
