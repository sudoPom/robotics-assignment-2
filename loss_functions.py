import numpy as np


class MSELoss:
    @staticmethod
    def loss(y, y_pred):
        return np.mean(np.power(y - y_pred, 2))

    @staticmethod
    def gradient(y, y_pred):
        return 2 * (y_pred - y)


class ISELoss:
    """Instantaneous Square Error Loss Function"""
    @staticmethod
    def loss(y, y_pred):
        """Computes the Instantaneous Square Error of the networks calculated values."""
        return 0.5 * np.sum(np.power(y - y_pred, 2))

    @staticmethod
    def gradient(y, y_pred):
        """Calculates the derivative of the sum of square errors"""
        return y_pred - y


class Softmax_CrossEntropyLoss:
    """Softmax Cross Entropy Function"""

    @staticmethod
    def loss(y, y_pred):
        """Computes the Softmax Cross Entropy on the networks calculated values."""
        return -np.sum(y * np.log(y_pred))

    @staticmethod
    def gradient(y, y_pred):
        return y_pred - y
