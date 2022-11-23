import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


class FullyConnectedLayer:
    # This is the class of the single layer of the fully connected neural network
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
        # calculate gradient of weights and bias based on the last layer's delta
        self.weights_grad = np.dot(self.input_data.T, self.activation.backward(delta))
        self.bias_grad = self.activation.backward(delta)

        # calculate delta for this layer and return it
        delta = np.dot(self.activation.backward(delta), self.weights.T)

        return delta

    def zero_grad(self):
        # clear gradient of weights and bias
        self.weights_grad = np.zeros_like(self.weights_grad)
        self.bias_grad = np.zeros_like(self.bias_grad)

        # clear the cached input data and output data
        self.input_data = None
        self.output_data = None


class NeuralNetwork:
    # This is the class of the whole neural network, which contains multiple layers
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

            if epoch % 50 == 0:  # print and record the loss of each 50 epochs
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
