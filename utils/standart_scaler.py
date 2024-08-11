import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0, ddof=0)
        self.scale_ = np.where(self.scale_ == 0, 1, self.scale_)

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("You must fit the scaler before transforming data.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("You must fit the scaler before inverse transforming data.")
        return (X * self.scale_) + self.mean_
