from network import NeuralNetwork
from activation_func import *
from loss_func import ISELoss
from optimizers import SGD

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


def load_and_predict(file_path: str):
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


def main():
    x = np.arange(-1, 1, 0.05)
    y = func(x)
    dataset = list(zip(x, y))

    model = NeuralNetwork()
    layer_config = [
        {'input_size': 1, 'output_size': 3, 'normalize': DirectLayer(), 'activation': Tanh()},
        {'input_size': 3, 'output_size': 1, 'normalize': DirectLayer(), 'activation': DirectLayer()},
    ]

    model.make_layers(layer_config, optimizer=SGD)
    model.train(dataset, epochs=10000, loss_func=ISELoss, lr=0.01)
    test_predict(model)
    model.dump_model('task1_model_SGD.pkl')


if __name__ == '__main__':
    main()
    # load_and_predict('task1_model_SGD.pkl')
