import numpy as np


class TSArm:
    def __init__(self, p):
        self.p = p
        self.a = 1
        self.b = 1

    def get_p_estimate(self):
        return self.a / (self.a + self.b)

    def get_outcome(self):
        # draw a 1 with probability p
        return np.random.random() < self.p

    def update(self, x):
        self.a += x
        self.b += 1 - x

    def sample(self):
        return np.random.beta(self.a, self.b)
