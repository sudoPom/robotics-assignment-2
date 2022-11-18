import numpy as np


class MSELoss:
    @staticmethod
    # calculate loss value
    def loss(y, y_pred):
        return 0.5 * np.mean(np.power(y - y_pred, 2))

    @staticmethod
    # calculate the gradient of the loss function
    def gradient(y, y_pred):
        return y_pred - y


class Softmax_CrossEntropyLoss:
    @staticmethod
    def loss(y, y_pred):
        return -np.sum(y * np.log(y_pred))

    @staticmethod
    def gradient(y, y_pred):
        return y_pred - y