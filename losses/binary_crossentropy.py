import numpy as np


class BinaryCrossentropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        self.y_pred = y_pred
        self.y_true = y_true
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self):
        return (self.y_pred - self.y_true) / (self.y_pred * (1 - self.y_pred)) / self.y_pred.shape[0]

    def accuracy(self, y_pred, y_true):
        y_pred_class = np.round(y_pred)
        correct_predictions = np.sum(y_pred_class == y_true)
        accuracy = correct_predictions / y_true.shape[0]
        return accuracy
