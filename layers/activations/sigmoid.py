import numpy as np


class ActivationSigmoid:
    def __init__(self):
        self.output = None
        self.inputs = None
        self.dinputs = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (self.output * (1 - self.output))
