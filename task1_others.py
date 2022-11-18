from network import NeuralNetwork
from activation_func import *
from loss_func import ISELoss
from optimizers import *

import os
import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 0.8 * np.power(x, 3) + 0.3 * np.power(x, 2) - 0.4 * x + np.random.normal(0, 0.02, x.shape)


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
    model.train(dataset, epochs=5000, loss_func=ISELoss, lr=0.01)

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
    model.train(dataset, epochs=5000, loss_func=ISELoss, lr=0.01)

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
    model.train(dataset, epochs=5000, loss_func=ISELoss, lr=0.01)

    # use the model to predict the result
    test_predict(model)
    model.dump_model('task1_model_others.pkl')


def train_model_different_lr():
    """
    Train the model with a different learning rate of 0.1 and the same SGD optimizer
    """
    x = np.arange(-1, 1, 0.05)
    y = func(x)
    dataset = list(zip(x, y))
    model = NeuralNetwork()

    # use the same 1-3-1 layer structure
    layer_config = [
        {'input_size': 1, 'output_size': 3, 'normalize': DirectLayer(), 'activation': Tanh()},
        {'input_size': 3, 'output_size': 1, 'normalize': DirectLayer(), 'activation': DirectLayer()},
    ]

    # use the same SGD optimizer to train the model
    model.make_layers(layer_config, optimizer=SGD)
    # use a larger learning rate of 0.1
    model.train(dataset, epochs=5000, loss_func=ISELoss, lr=0.1)

    # use the model to predict the result
    test_predict(model)
    model.dump_model('task1_model_others.pkl')


if __name__ == '__main__':
    train_model_adam_optimizer()
    # train_model_different_size()
    # train_model_different_activation()
    # train_model_different_lr()

    # load_and_predict('task1_model_others.pkl')
