import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 0.8 * np.power(x, 3) + 0.3 * np.power(x, 2) - 0.4 * x + np.random.normal(0, 0.02, x.shape)


class ReLu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        grad * (self.x > 0)
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Leaky_ReLu:
    def __init__(self, alpha):
        self.x = None
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.maximum(self.alpha * x, x)

    def backward(self, grad):
        return grad * (self.x > 0) + self.alpha * grad * (self.x <= 0)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Tanh:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.tanh(x)

    def backward(self, grad):
        return grad * (1 - np.tanh(self.x) ** 2)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class DirectLayer:
    def forward(self, x):
        return x

    def backward(self, grad):
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MSELoss:
    @staticmethod
    def loss(y, y_pred):
        return 0.5 * np.sum(np.power(y - y_pred, 2))

    @staticmethod
    def gradient(y, y_pred):
        return y_pred - y


class Normalization:
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


class Adam:
    def __init__(self, beta1=0.999, beta2=0.999, eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.t = 0

    def update(self, grad):
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.t = min(self.t + 1, 100000)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        m_hat = self.m / (1 - np.power(self.beta1, self.t))

        self.v = self.beta2 * self.v + (1 - self.beta2) * np.power(grad, 2)
        v_hat = self.v / (1 - np.power(self.beta2, self.t))

        return m_hat / (np.sqrt(v_hat) + self.eps)

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)


class SGD:
    def __init__(self):
        pass

    def update(self, grad):
        return grad

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)


class FullyConnectedLayer:
    def __init__(self, config: dict):
        self.normalize = config['normalize']
        self.activation = config['activation']
        input_size = config['input_size']
        output_size = config['output_size']

        self.input_data = None
        self.output_data = None
        self.weights_grad = np.zeros((input_size, output_size))
        self.bias_grad = np.zeros((1, output_size))
        self.weights = np.random.uniform(-1, 1, (input_size, output_size))
        self.bias = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, input_data):
        self.input_data = np.array(input_data)
        self.output_data = np.dot(input_data, self.weights) + self.bias
        self.output_data = self.activation(self.normalize(self.output_data))

        return self.output_data

    def backward(self, delta):
        self.weights_grad = self.activation.backward(
            np.dot(self.input_data.T, delta))
        self.bias_grad = self.activation.backward(delta)
        delta = np.dot(delta, self.weights.T)

        return delta

    def zero_grad(self):
        self.weights_grad = np.zeros_like(self.weights_grad)
        self.bias_grad = np.zeros_like(self.bias_grad)
        self.input_data = None
        self.output_data = None


class NeuralNetwork:
    def __init__(self):
        self.modules = []

    def make_layers(self, layer_configs, optimizer):
        for config in layer_configs:
            optim_w = optimizer()
            optim_b = optimizer()
            self.modules.append(
                (FullyConnectedLayer(config), optim_w, optim_b))

    def forward(self, input_data):
        for module, _, _ in self.modules:
            input_data = module.forward(input_data)

        return input_data

    def backward(self, delta):
        for module, _, _ in reversed(self.modules):
            delta = module.backward(delta)

    def update(self, lr):
        for module, optim_w, optim_b in self.modules:
            module.weights -= lr * optim_w(module.weights_grad)
            module.bias -= lr * optim_b(module.bias_grad)
            module.zero_grad()

    def train(self, dataset, epochs, loss_func, lr):
        for epoch in range(epochs):
            loss_list = []
            np.random.shuffle(dataset)
            for data_x, data_y in dataset:
                y_hat = self.forward(np.array(data_x))
                loss = loss_func.loss(np.array(data_y), y_hat)
                loss_list.append(loss)
                delta = loss_func.gradient(np.array(data_y), y_hat)
                self.backward(delta)
                self.update(lr)

            if epoch % 100 == 0:
                print('Epoch: {}, Loss: {}'.format(epoch, np.mean(loss_list)))

    def predict(self, x):
        return self.forward(x)

    def dump_model(self, path):
        model_dict = {'modules': self.modules}
        with open(path, 'wb') as f:
            pickle.dump(model_dict, f)

    def load_model(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError('File not found: {}'.format(path))

        with open(path, 'rb') as f:
            model_dict = pickle.load(f)

        self.modules = model_dict['modules']


def main():
    x = np.arange(-1, 1, 0.05)
    y = func(x)
    dataset = list(zip(x, y))

    Net = NeuralNetwork()
    layer_config = [
        {'input_size': 1, 'output_size': 3,
            'normalize': DirectLayer(), 'activation': Tanh()},
        {'input_size': 3, 'output_size': 1,
            'normalize': DirectLayer(), 'activation': DirectLayer()},
    ]

    Net.make_layers(layer_config, optimizer=Adam)
    Net.train(dataset, epochs=5000, loss_func=MSELoss, lr=0.002)

    y_pred = []
    for data_x in x:
        y_pred.append(Net.predict(data_x).item())

    plt.scatter(x, y, label='y_dataset')
    plt.plot(x, y_pred, label='y_pred', color='red')
    plt.legend()
    plt.show()
    selection = ''
    while selection.upper() not in ['N', 'Y']:
        selection = input('Do you want to save the model(Y/N): ')
    if selection.upper() == 'Y':
        Net.dump_model('model_task1.pkl')


def test_predict():
    Net = NeuralNetwork()
    Net.load_model('model.pkl')
    x = np.arange(-0.97, 0.93, 0.02)
    y = func(x)
    y_pred = []
    for data_x in x:
        y_pred.append(Net.predict(data_x).item())
    plt.scatter(x, y, label='y_dataset')
    plt.plot(x, y_pred, label='y_pred', color='red')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
