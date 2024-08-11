import numpy as np


class ActivationSoftmax:
    def __init__(self):
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        e_x = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        batch_size = self.dinputs.shape[0]
        self.dinputs[range(batch_size), np.argmax(self.output, axis=1)] -= 1
        self.dinputs /= batch_size
        return self.dinputs
