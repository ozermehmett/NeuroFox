class ActivationLinear:
    def __init__(self):
        self.output = None
        self.dinputs = None

    def forward(self, inputs):
        self.output = inputs

    def backward(self, dvalues):
        self.dinputs = dvalues
