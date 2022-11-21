from network import NeuralNetwork
from activation_func import *
from loss_func import *
from optimizers import *
import numpy as np

# Transform the label to one-hot encoding
one_hot_encoding = {
    'Iris-setosa': np.array([1, 0, 0]),
    'Iris-versicolor': np.array([0, 1, 0]),
    'Iris-virginica': np.array([0, 0, 1])
}

result_label = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']


# Decoding the one-hot encoding to the number
def one_hot_decode(one_hot):
    return np.argmax(one_hot)


def prep_dataset(file_path: str, split_ratio=0.7):
    """
    Prepare the dataset for training and validation

    Parameters
    ----------
    file_path: str
        The path of the dataset
    split_ratio: float
        The ratio of the training set to the whole dataset
    """
    dataset = []
    with open(file_path) as f:
        for line in f:
            data = line.strip().split(',')
            data_x = np.array(data[:4])
            data_x = data_x.astype(np.float_)
            data_x = data_x[np.newaxis, :]
            data_y = one_hot_encoding[data[-1]]
            dataset.append((data_x, data_y))

    # shuffle the dataset
    np.random.shuffle(dataset)

    # split the dataset into training set and validation set with the split ratio
    split_idx = round(len(dataset) * split_ratio)
    training_set = dataset[:split_idx]
    validation_set = dataset[split_idx:]

    return training_set, validation_set


def validate(model, validation_set):
    """
    Validate the model with the validation set

    Parameters
    ----------
    model: NeuralNetwork
        The model to be validated
    validation_set: list
        The validation set to validate the model
    """
    correct = 0
    for data_x, data_y in validation_set:
        y_hat = model.predict(data_x)
        y_class = one_hot_decode(y_hat)
        data_y_class = one_hot_decode(data_y)
        if y_class == data_y_class:
            correct += 1

    return correct / len(validation_set)


def test_predict(path: str):
    """
    Test a trained model with a single input

    Parameters
    ----------
    path: str
        The path of the model
    """
    Net = NeuralNetwork()
    Net.load_model(path)

    # set the input data to get its prediction
    test_data = np.array([[5.8, 2.8, 5.1, 2.4]])
    prediction = Net.predict(test_data)

    # print the raw prediction
    print(prediction)

    # print the classification
    classification = np.argmax(prediction)
    print(classification)

    # print the label of the classification
    label = result_label[classification]
    print(label)


def train_model_adam_optimizer():
    train_set, validation_set = prep_dataset('IrisData.txt')
    model = NeuralNetwork()
    layer_config = [
        {'input_size': 4, 'output_size': 5, 'normalize': Normalization(), 'activation': Tanh()},
        {'input_size': 5, 'output_size': 3, 'normalize': Normalization(), 'activation': Tanh()},
        {'input_size': 3, 'output_size': 3, 'normalize': DirectLayer(), 'activation': Softmax()},
    ]

    model.make_layers(layer_config, optimizer=Adam)
    model.train(train_set, epochs=5000, loss_func=Softmax_CrossEntropyLoss, lr=0.002)
    accuracy = validate(model, validation_set)
    print('Accuracy: {}'.format(accuracy))
    model.dump_model('task2_model_others.pkl')


def train_model_different_activation():
    train_set, validation_set = prep_dataset('IrisData.txt')
    model = NeuralNetwork()
    layer_config = [
        {'input_size': 4, 'output_size': 5, 'normalize': DirectLayer(), 'activation': ReLu()},
        {'input_size': 5, 'output_size': 3, 'normalize': DirectLayer(), 'activation': ReLu()},
        {'input_size': 3, 'output_size': 3, 'normalize': DirectLayer(), 'activation': Softmax()},
    ]

    model.make_layers(layer_config, optimizer=SGD)
    model.train(train_set, epochs=5000, loss_func=Softmax_CrossEntropyLoss, lr=0.001)
    accuracy = validate(model, validation_set)
    print('Accuracy: {}'.format(accuracy))
    model.dump_model('task2_model_others.pkl')


def train_model_different_size():
    train_set, validation_set = prep_dataset('IrisData.txt')
    model = NeuralNetwork()
    layer_config = [
        {'input_size': 4, 'output_size': 8, 'normalize': Normalization(), 'activation': Tanh()},
        {'input_size': 8, 'output_size': 5, 'normalize': Normalization(), 'activation': Tanh()},
        {'input_size': 5, 'output_size': 3, 'normalize': DirectLayer(), 'activation': Softmax()},
    ]

    model.make_layers(layer_config, optimizer=SGD)
    model.train(train_set, epochs=5000, loss_func=Softmax_CrossEntropyLoss, lr=0.005)
    accuracy = validate(model, validation_set)
    print('Accuracy: {}'.format(accuracy))
    model.dump_model('task2_model_others.pkl')


def train_model_different_lr():
    train_set, validation_set = prep_dataset('IrisData.txt')
    model = NeuralNetwork()
    layer_config = [
        {'input_size': 4, 'output_size': 5, 'normalize': Normalization(), 'activation': Tanh()},
        {'input_size': 5, 'output_size': 3, 'normalize': Normalization(), 'activation': Tanh()},
        {'input_size': 3, 'output_size': 3, 'normalize': DirectLayer(), 'activation': Softmax()},
    ]

    model.make_layers(layer_config, optimizer=SGD)
    model.train(train_set, epochs=5000, loss_func=Softmax_CrossEntropyLoss, lr=0.01)
    accuracy = validate(model, validation_set)
    print('Accuracy: {}'.format(accuracy))
    model.dump_model('task2_model_others.pkl')


if __name__ == '__main__':
    train_model_adam_optimizer()
    # train_model_different_activation()
    # train_model_different_size()
    # train_model_different_lr()

    # test_predict('task2_model_others.pkl')
