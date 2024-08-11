import numpy as np


class BinaryFocalLoss:
    def __init__(self, gamma=2, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
        self.y_pred = None
        self.y_true = None

    def forward(self, y_pred, y_true):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        self.y_pred = y_pred
        self.y_true = y_true
        p_t = self.y_true * self.y_pred + (1 - self.y_true) * (1 - self.y_pred)
        focal_loss = -self.alpha * (1 - p_t) ** self.gamma * np.log(p_t)
        return np.mean(focal_loss)

    def backward(self):
        epsilon = 1e-15
        y_pred = np.clip(self.y_pred, epsilon, 1 - epsilon)
        p_t = self.y_true * y_pred + (1 - self.y_true) * (1 - y_pred)
        grad = -self.alpha * (1 - p_t) ** self.gamma * (self.y_true - y_pred) / p_t
        return grad / self.y_true.size

    def accuracy(self, y_pred, y_true):
        y_pred_class = np.round(y_pred)
        correct_predictions = np.sum(y_pred_class == y_true)
        accuracy = correct_predictions / y_true.shape[0]
        return accuracy
