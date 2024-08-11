import numpy as np


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, layer):
        if layer not in self.m:
            self.m[layer] = {'weights': np.zeros_like(layer.weights), 'biases': np.zeros_like(layer.biases)}
            self.v[layer] = {'weights': np.zeros_like(layer.weights), 'biases': np.zeros_like(layer.biases)}

        self.t += 1
        m = self.m[layer]
        v = self.v[layer]

        m['weights'] = self.beta1 * m['weights'] + (1 - self.beta1) * layer.dweights
        m['biases'] = self.beta1 * m['biases'] + (1 - self.beta1) * layer.dbiases

        v['weights'] = self.beta2 * v['weights'] + (1 - self.beta2) * (layer.dweights ** 2)
        v['biases'] = self.beta2 * v['biases'] + (1 - self.beta2) * (layer.dbiases ** 2)

        m_hat_weights = m['weights'] / (1 - self.beta1 ** self.t)
        m_hat_biases = m['biases'] / (1 - self.beta1 ** self.t)

        v_hat_weights = v['weights'] / (1 - self.beta2 ** self.t)
        v_hat_biases = v['biases'] / (1 - self.beta2 ** self.t)

        layer.weights -= self.learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        layer.biases -= self.learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)
