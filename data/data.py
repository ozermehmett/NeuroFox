import numpy as np


def create_xor_data(num_samples=100):
    np.random.seed(42)
    X = np.random.randint(0, 2, size=(num_samples, 2))
    y = np.bitwise_xor(X[:, 0], X[:, 1])
    y = y.reshape(-1, 1)
    return X, y


def create_binary_classification_data(num_samples=100):
    np.random.seed(42)
    X = np.random.rand(num_samples, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int).reshape(-1, 1)
    return X, y
