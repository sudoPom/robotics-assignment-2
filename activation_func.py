import numpy as np


class ReLu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    # calculate the gradient of ReLu
    def backward(self, grad):
        grad * (self.x > 0)
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Tanh:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, grad):
        # calculate the gradient of Tanh
        return grad * (1 - np.tanh(self.x) ** 2)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class DirectLayer:
    # This is the direct layer, which is used to replace the activation function if no activation is needed
    def forward(self, x):
        return x

    def backward(self, grad):
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Softmax:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, grad):
        """
        I combine the gradient of softmax and cross_entropy loss together in the cross_entropy loss gradient
        so the gradient of softmax is 1, the joint gradient is not calculated here
        The combination gradient is much simpler and more efficient than calculating the gradient separately
        reference to https://deepnotes.io/softmax-crossentropy
        """
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Normalization:
    # Layer normalization is used to normalize the input of each layer
    def __init__(self):
        self.x = None
        self.mean = None
        self.std = None

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x)
        self.std = np.std(x)
        return (x - self.mean) / self.std

    def backward(self, grad):
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
