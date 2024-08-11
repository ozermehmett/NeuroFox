import numpy as np


class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, inputs, training=True):
        if not training:
            return inputs

        # Create dropout mask
        self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape)
        return inputs * self.mask / (1 - self.rate)

    def backward(self, dvalues):
        return dvalues * self.mask / (1 - self.rate)
