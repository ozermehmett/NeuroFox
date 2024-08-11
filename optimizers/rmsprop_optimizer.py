import numpy as np


class RMSpropOptimizer:
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}

    def update(self, layer):
        if layer not in self.cache:
            self.cache[layer] = {'weights': np.zeros_like(layer.weights), 'biases': np.zeros_like(layer.biases)}

        self.cache[layer]['weights'] = self.beta * self.cache[layer]['weights'] + (1 - self.beta) * layer.dweights ** 2
        self.cache[layer]['biases'] = self.beta * self.cache[layer]['biases'] + (1 - self.beta) * layer.dbiases ** 2

        layer.weights -= self.learning_rate * layer.dweights / (np.sqrt(self.cache[layer]['weights']) + self.epsilon)
        layer.biases -= self.learning_rate * layer.dbiases / (np.sqrt(self.cache[layer]['biases']) + self.epsilon)
