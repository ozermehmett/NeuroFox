import numpy as np


class CategoricalCrossentropy:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        self.y_pred = y_pred
        self.y_true = y_true
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def backward(self):
        epsilon = 1e-15
        y_pred = np.clip(self.y_pred, epsilon, 1. - epsilon)
        grad = - (self.y_true / y_pred) / self.y_true.shape[0]
        return grad

    def accuracy(self, y_pred, y_true):
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(y_true, axis=1)
        correct_predictions = np.sum(y_pred_class == y_true_class)
        accuracy = correct_predictions / y_true.shape[0]
        return accuracy
