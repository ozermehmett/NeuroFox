import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    num_samples = X.shape[0]

    num_test_samples = int(num_samples * test_size)

    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, X_test, y_train, y_test
