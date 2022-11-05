import numpy as np
import matplotlib.pyplot as plt
import copy


def func(x):
    return 0.8 * np.power(x, 3) + 0.3 * np.power(x, 2) - 0.4 * x + np.random.normal(0, 0.02)


class ReLu:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad):
        return grad * (self.x > 0)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Leaky_ReLu:
    def __init__(self, alpha=0.01):
        self.x = None
        self.alpha = alpha

    def forward(self, x):
        self.x = x
        return np.maximum(self.alpha * x, x)

    def backward(self, grad):
        return grad * (self.x > 0) + self.alpha * grad * (self.x <= 0)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class DirectLayer:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def backward(self, grad):
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MSELoss:
    def __init__(self, y, y_hat):
        self.y = y  # real value
        self.y_hat = y_hat  # predict data

    def loss(self):
        return np.mean(np.power(self.y - self.y_hat, 2))

    def gradient(self):
        return 2 * (self.y_hat - self.y)


class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation=Leaky_ReLu, output_layer=False):
        self.weights = np.random.normal(0, 0.01, (input_size, output_size))
        self.bias = np.random.normal(0, 0.01, (1, output_size))
        self.activation = DirectLayer()
        self.input_data = None
        self.output_data = None
        self.weights_grad = np.zeros((input_size, output_size))
        self.bias_grad = np.zeros((1, output_size))
        if not output_layer:
            self.activation = activation()

    def forward(self, input_data):
        self.input_data = input_data
        self.output_data = np.dot(input_data, self.weights) + self.bias
        self.output_data = self.activation(self.output_data)

        return self.output_data

    def backward(self, delta):
        self.weights_grad = self.activation.backward(np.dot(self.input_data.T, delta))
        self.bias_grad = self.activation.backward(delta)
        delta = np.dot(delta, self.activation.backward(self.weights.T))

        return delta

    def zero_grad(self):
        self.weights_grad = np.zeros_like(self.weights_grad)
        self.bias_grad = np.zeros_like(self.bias_grad)
        self.input_data = None
        self.output_data = None


class NeuralNetwork:
    def __init__(self, layer_configurations):
        self.modules = []
        self.layer_configurations = layer_configurations
        self.make_layers(layer_configurations)

    def make_layers(self, layer_config):
        for input_size, output_size in layer_config[:-1]:
            self.modules.append(FullyConnectedLayer(input_size, output_size))

        input_size, output_size = layer_config[-1]
        self.modules.append(FullyConnectedLayer(input_size, output_size, output_layer=True))

    def forward(self, input_data):
        data = copy.deepcopy(input_data)
        for module in self.modules:
            data = module.forward(data)

        return data

    def backward(self, delta):
        for module in reversed(self.modules):
            delta = module.backward(delta)

    def update(self, lr):
        for module in self.modules:
            module.weights -= lr * module.weights_grad
            module.bias -= lr * module.bias_grad
            module.zero_grad()

    def train(self, dataset, epochs, lr):
        for epoch in range(epochs):
            loss_list = []
            np.random.shuffle(dataset)
            for data_x, data_y in dataset:
                y_hat = self.forward(data_x)
                loss = MSELoss(data_y, y_hat)
                loss_list.append(loss.loss())
                delta = loss.gradient()
                self.backward(delta)
                self.update(lr)

            if epoch % 100 == 0:
                print('Epoch: {}, Loss: {}'.format(epoch, np.mean(loss_list)))

    def predict(self, x):
        return self.forward(x)


def main():
    x = np.linspace(-2, 2, 100)
    y = func(x)
    dataset = list(zip(x, y))

    layer_config = [(1, 3), (3, 1)]
    Net = NeuralNetwork(layer_config)
    Net.train(dataset, epochs=5000, lr=0.001)
    y_pred = []
    for data_x in x:
        y_pred.append(Net.predict(data_x).item())

    plt.scatter(x, y, label='y_dataset')
    plt.plot(x, y_pred, label='y_pred', color='red')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
