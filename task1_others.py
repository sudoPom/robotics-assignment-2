import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 0.8 * np.power(x, 3) + 0.3 * np.power(x, 2) - 0.4 * x + np.random.normal(0, 0.02, x.shape)


class Tanh:
    def __init__(self):
        self.x = None

    def forward(self, x):
        # Save the input for the backward pass
        self.x = x
        return np.tanh(x)

    def backward(self, grad):
        # The gradient of tanh is (1 - tanh(x)^2)
        return grad * (1 - np.tanh(self.x) ** 2)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


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


class DirectLayer:
    def forward(self, x):
        return x

    def backward(self, grad):
        return grad

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MSELoss:
    @staticmethod
    # calculate loss value
    def loss(y, y_pred):
        return 0.5 * np.mean(np.power(y - y_pred, 2))

    @staticmethod
    # calculate the gradient of the loss function
    def gradient(y, y_pred):
        return y_pred - y


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


def test_predict(model):
    """
    Plot the prediction result after the training process

    Parameters
    ----------
    model: NeuralNetwork
        The trained model used to predict the result

    """
    # set the range of x
    x = np.arange(-0.97, 0.93, 0.02)
    # get its true value directly from the function
    y = func(x)
    # set a list to store the predicted value
    y_pred = []
    for data_x in x:
        # get the predicted value of each x
        predict = model.predict(data_x).item()

        # append it to the list
        y_pred.append(predict)

    # plot the true value and the predicted value
    plt.scatter(x, y, label='y_dataset')
    plt.plot(x, y_pred, label='y_pred', color='red')
    plt.title('Ground Truth and Prediction Result')
    plt.legend()
    plt.show()


class FullyConnectedLayer:
    def __init__(self, config: dict):
        # set the normalization and activation function based on the config
        self.normalize = config['normalize']
        self.activation = config['activation']
        input_size = config['input_size']
        output_size = config['output_size']

        # set the variable to cache input data and output data
        self.input_data = None
        self.output_data = None
        self.weights_grad = np.zeros((input_size, output_size))
        self.bias_grad = np.zeros((1, output_size))

        # initialize weights and bias with random values
        self.weights = np.random.uniform(-1, 1, (input_size, output_size))
        self.bias = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, input_data):
        # cache input data
        self.input_data = np.array(input_data)

        # calculate output data
        self.output_data = np.dot(input_data, self.weights) + self.bias

        # apply activation function to output data
        self.output_data = self.activation(self.normalize(self.output_data))

        return self.output_data

    def backward(self, delta):
        # calculate gradient of weights and bias based on the previous layer's delta
        self.weights_grad = self.activation.backward(np.dot(self.input_data.T, delta))
        self.bias_grad = self.activation.backward(delta)

        # calculate delta for this layer and return it
        delta = np.dot(delta, self.weights.T)

        return delta

    def zero_grad(self):
        # clear gradient of weights and bias
        self.weights_grad = np.zeros_like(self.weights_grad)
        self.bias_grad = np.zeros_like(self.bias_grad)

        # clear the cached input data and output data
        self.input_data = None
        self.output_data = None


class NeuralNetwork:
    def __init__(self):
        # list of modules in the network
        self.modules = []

    def make_layers(self, layer_configs, optimizer):
        # make layers based on the layer configs
        for config in layer_configs:
            # create a new layer with its own optimizer
            optim_w = optimizer()
            optim_b = optimizer()

            # store the layer and its optimizer to the module list
            self.modules.append((FullyConnectedLayer(config), optim_w, optim_b))

    def forward(self, input_data):
        # forward the input data through the network
        for module, _, _ in self.modules:
            input_data = module.forward(input_data)

        return input_data

    def backward(self, delta):
        # backward the delta through the network
        for module, _, _ in reversed(self.modules):
            delta = module.backward(delta)

    def update(self, lr):
        # update the weights and bias of each layer since the gradient of weights and bias has been calculated
        for module, optim_w, optim_b in self.modules:
            # update weights and bias with the optimizer and learning rate
            module.weights += optim_w(lr, module.weights_grad)
            module.bias += optim_b(lr, module.bias_grad)
            # clear the gradient of weights and bias after updating
            module.zero_grad()

    def train(self, dataset, epochs, loss_func, lr):
        loss_list = []
        for epoch in range(epochs):
            # record the loss of each epoch
            loss_epoch = []
            # shuffle the dataset in each epoch to avoid overfitting
            np.random.shuffle(dataset)
            for data_x, data_y in dataset:
                # forward the input data through the network
                y_hat = self.forward(np.array(data_x))
                # calculate the loss of the output data and the ground truth
                loss = loss_func.loss(np.array(data_y), y_hat)
                # append the loss value to the loss list
                loss_epoch.append(loss)
                # calculate the delta of the output layer
                delta = loss_func.gradient(np.array(data_y), y_hat)
                # backward the delta through the network
                self.backward(delta)
                # update the weights and bias of each layer
                self.update(lr)

            if epoch % 100 == 0:  # print and record the loss of each 100 epochs
                loss_list.append((epoch, np.mean(loss_epoch)))
                print('Epoch: {}, Loss: {}'.format(epoch, np.mean(loss_epoch)))

        self.plot_loss(loss_list)

    def plot_loss(self, loss_list):
        # plot the loss curve
        plt.plot([x[0] for x in loss_list], [x[1] for x in loss_list])
        plt.title('The Training Loss versus Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def predict(self, x):
        # forward the input data through the network
        return self.forward(x)

    def dump_model(self, path):
        # dump the model to the path
        model_dict = {'modules': self.modules}
        with open(path, 'wb') as f:
            pickle.dump(model_dict, f)

    def load_model(self, path):
        # load the model from the path
        if not os.path.isfile(path):
            raise FileNotFoundError('File not found: {}'.format(path))

        with open(path, 'rb') as f:
            model_dict = pickle.load(f)

        # load the modules from the model dict
        self.modules = model_dict['modules']


def load_and_predict(file_path):
    """
    Load the trained model and plot the prediction result directly

    Parameters
    ----------
    file_path: str
        The path of the model file

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError('The file does not exist')

    model = NeuralNetwork()
    model.load_model(file_path)
    test_predict(model)


def train_model_adam_optimizer():
    """
    Train the model with the same network structure and Adam optimizer
    """
    x = np.arange(-1, 1, 0.05)
    y = func(x)
    dataset = list(zip(x, y))
    model = NeuralNetwork()

    # keep the same 1-3-1 layer structure
    layer_config = [
        {'input_size': 1, 'output_size': 3, 'normalize': DirectLayer(), 'activation': Tanh()},
        {'input_size': 3, 'output_size': 1, 'normalize': DirectLayer(), 'activation': DirectLayer()},
    ]

    # use Adam optimizer to train the model
    model.make_layers(layer_config, optimizer=Adam)
    model.train(dataset, epochs=5000, loss_func=MSELoss, lr=0.01)

    # use the model to predict the result
    test_predict(model)
    model.dump_model('task1_model_others.pkl')


def train_model_different_size():
    """
    Train the model with a larger network of 1-10-1 and the same SGD optimizer
    """
    x = np.arange(-1, 1, 0.05)
    y = func(x)
    dataset = list(zip(x, y))
    model = NeuralNetwork()

    # use larger network structure with 1-10-1
    layer_config = [
        {'input_size': 1, 'output_size': 10, 'normalize': DirectLayer(), 'activation': Tanh()},
        {'input_size': 10, 'output_size': 1, 'normalize': DirectLayer(), 'activation': DirectLayer()},
    ]

    # use the same SGD optimizer to train the model
    model.make_layers(layer_config, optimizer=SGD)
    model.train(dataset, epochs=5000, loss_func=MSELoss, lr=0.01)

    # use the model to predict the result
    test_predict(model)
    model.dump_model('task1_model_others.pkl')


def train_model_different_activation():
    """
    Train the model with a different activation function of ReLU and the same SGD optimizer
    """
    x = np.arange(-1, 1, 0.05)
    y = func(x)
    dataset = list(zip(x, y))
    model = NeuralNetwork()

    # use the same 1-3-1 layer structure but use different activation function
    layer_config = [
        # use the ReLU activation function instead of Tanh
        {'input_size': 1, 'output_size': 3, 'normalize': DirectLayer(), 'activation': ReLu()},
        {'input_size': 3, 'output_size': 1, 'normalize': DirectLayer(), 'activation': DirectLayer()},
    ]

    # use the same SGD optimizer to train the model
    model.make_layers(layer_config, optimizer=SGD)
    model.train(dataset, epochs=5000, loss_func=MSELoss, lr=0.01)

    # use the model to predict the result
    test_predict(model)
    model.dump_model('task1_model_others.pkl')


if __name__ == '__main__':
    train_model_adam_optimizer()
    # train_model_different_size()
    # train_model_different_activation()
    # load_and_predict('task1_model_others.pkl')
