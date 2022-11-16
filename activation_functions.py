import numpy as np


class Sigmoid:
    """Sigmoid Activation Function

    Attributes:
        x: Input to the function.    
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        """Calculates the result of the function."""
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, grad):
        """Calculates the derivative of the function and scales the input gradient by it"""
        return grad * (self.x * (1 - self.x))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Softmax:
    """Softmax Activation Function.

    Attributes:
        x: Input to the function.    
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        """Calculates the result of the function."""
        self.x = x
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, grad):
        """Calculates the derivative of the function and scales the input gradient by it"""
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class ReLu:
    """ReLu Activation Function.

    Attributes:
        x: Input to the function.
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        """Calculates the result of the function on its input."""
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        """Returns the input gradient scaled by the derivative of the function at the point of it's input."""
        grad * (self.x > 0)
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Leaky_ReLu:
    """Leaky ReLu Activation Function

    Attributes:
        x: Input to the function.
        alpha: Value to scale negative inputs by.
    """

    def __init__(self, alpha):
        self.x = None
        self.alpha = alpha

    def forward(self, x):
        """Calculates the result of the function on its input."""
        self.x = x
        return np.maximum(self.alpha * x, x)

    def backward(self, grad):
        """Returns the input gradient scaled by the derivative of the function at the point of it's input."""
        return grad * (self.x > 0) + self.alpha * grad * (self.x <= 0)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Tanh:
    """Tanh Activation Function

    Attributes:
        x: Input to the function.
    """

    def __init__(self):
        self.x = None

    def forward(self, x):
        """Calculates the result of the function on its input."""
        self.x = x
        return np.tanh(x)

    def backward(self, grad):
        """Returns the input gradient scaled by the derivative of the function at the point of it's input."""
        return grad * (1 - np.tanh(self.x) ** 2)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class DirectLayer:
    def forward(self, x):
        """Calculates the result of the function on its input."""
        return x

    def backward(self, grad):
        """Returns the input gradient scaled by the derivative of the function at the point of it's input."""
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
