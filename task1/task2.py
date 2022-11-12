import os
import numpy as np
import pickle

one_hot_encoding = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}

result_label = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


class Sigmoid:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = x
        return 1 / (1 + np.exp(-x))

    def backward(self, grad):
        return grad * (self.x * (1 - self.x))

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
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


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
        return np.mean(np.power(y - y_pred, 2))

    @staticmethod
    def gradient(y, y_pred):
        return 2 * (y_pred - y)


class CrossEntropyLoss:
    @staticmethod
    def loss(y, y_pred):
        return -np.sum(y * np.log(y_pred))

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
        self.weights_grad = self.activation.backward(np.dot(self.input_data.T, delta))
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
            self.modules.append((FullyConnectedLayer(config), optim_w, optim_b))

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


def prep_dataset(file_path: str):
    dataset = []
    with open(file_path) as f:
        for line in f:
            data = line.strip().split(',')
            data_x = np.array(data[:4])
            data_x = data_x.astype(np.float_)
            data_x = data_x[np.newaxis, :]
            data_y = one_hot_encoding[data[-1]]
            data_y = data_y[np.newaxis, :]
            dataset.append((data_x, data_y))

    return dataset


def main():
    dataset = prep_dataset('IrisData.txt')
    Net = NeuralNetwork()
    layer_config = [
        {'input_size': 4, 'output_size': 10, 'normalize': Normalization(), 'activation': Tanh()},
        {'input_size': 10, 'output_size': 3, 'normalize': DirectLayer(), 'activation': Softmax()},
    ]

    Net.make_layers(layer_config, optimizer=Adam)
    Net.train(dataset, epochs=5000, loss_func=MSELoss, lr=0.002)
    Net.dump_model('model_task2.pkl')


def test_predict():
    Net = NeuralNetwork()
    Net.load_model('model_task2.pkl')
    test_data = np.array([[5.9,1.5,5.1,1.8]])
    prediction = Net.predict(test_data)
    print(prediction)
    classification = np.argmax(prediction)
    print(classification)
    label = result_label[classification]
    print(label)


if __name__ == '__main__':
    test_predict()
