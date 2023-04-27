import numpy as np

class EGArm:

    def __init__(self, p):
        self.p = p
        self.p_estimate = 0.5
        self.N = 0
        self.Ns = 0

    def get_p_estimate(self):
        return self.p_estimate

    def get_outcome(self):
        return np.random.random() < self.p

    def update(self, x):
        self.Ns += x
        self.N += 1
        self.p_estimate = self.Ns / self.N

    def sample(self):
        return self.p_estimate
