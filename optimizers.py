import numpy as np


class SGD:
    def __init__(self):
        pass

    def update(self, lr, grad):
        return -lr * grad

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)


class Adam:
    def __init__(self, beta1=0.999, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, lr, grad):
        # initialize m and v with 0 if they are None
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.t = min(self.t + 1, 100000)  # prevent overflow

        # update m and v
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        m_hat = self.m / (1 - np.power(self.beta1, self.t))

        # update v
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)
        v_hat = self.v / (1 - np.power(self.beta2, self.t))

        return -lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)


class AdaGrad:
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.v = None

    def update(self, lr, grad):
        # initialize v
        if self.v is None:
            self.v = np.zeros_like(grad)

        # update v
        self.v += np.power(grad, 2)

        return -lr * grad / (np.sqrt(self.v) + self.eps)

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)


class RMSProp:
    def __init__(self, beta=0.9, eps=1e-8):
        self.beta = beta
        self.eps = eps
        self.v = None

    def update(self, lr, grad):
        # initialize v
        if self.v is None:
            self.v = np.zeros_like(grad)

        # update v
        self.v = self.beta * self.v + (1 - self.beta) * np.power(grad, 2)

        return -lr * grad / (np.sqrt(self.v) + self.eps)

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)
