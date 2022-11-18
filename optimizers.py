import numpy as np


class Adam:
    def __init__(self, beta1=0.999, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, lr, grad):
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.t = min(self.t + 1, 100000)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        m_hat = self.m / (1 - np.power(self.beta1, self.t))

        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)
        v_hat = self.v / (1 - np.power(self.beta2, self.t))

        return -lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)


class SGD:
    def __init__(self):
        pass

    def update(self, lr, grad):
        return -lr * grad

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)
