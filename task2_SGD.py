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


# Decoding the one-hot encoding to the number of the class
def one_hot_decode(one_hot):
    return np.argmax(one_hot)


def prep_dataset(file_path: str, split_ratio=0.7):
    dataset = []
    with open(file_path) as f:
        for line in f:
            data = line.strip().split(',')
            data_x = np.array(data[:4])
            data_x = data_x.astype(np.float_)
            data_x = data_x[np.newaxis, :]
            data_y = one_hot_encoding[data[-1]]
            dataset.append((data_x, data_y))

    np.random.shuffle(dataset)
    split_idx = round(len(dataset) * split_ratio)
    training_set = dataset[:split_idx]
    validation_set = dataset[split_idx:]

    return training_set, validation_set


def validate(model, validation_set):
    correct = 0
    for data_x, data_y in validation_set:
        y_hat = model.predict(data_x)
        y_class = one_hot_decode(y_hat)
        data_y_class = one_hot_decode(data_y)
        if y_class == data_y_class:
            correct += 1

    return correct / len(validation_set)


def test_predict():
    Net = NeuralNetwork()
    Net.load_model('task2_model_SGD.pkl')
    test_data = np.array([[4.9, 3.1, 1, 0.1]])
    prediction = Net.predict(test_data)
    print(prediction)
    classification = np.argmax(prediction)
    print(classification)
    label = result_label[classification]
    print(label)


def main():
    train_set, test_set = prep_dataset('IrisData.txt')
    model = NeuralNetwork()
    layer_config = [
        {'input_size': 4, 'output_size': 5, 'normalize': Normalization(), 'activation': Tanh()},
        {'input_size': 5, 'output_size': 3, 'normalize': Normalization(), 'activation': Tanh()},
        {'input_size': 3, 'output_size': 3, 'normalize': DirectLayer(), 'activation': Softmax()},
    ]

    model.make_layers(layer_config, optimizer=SGD)
    model.train(train_set, epochs=5000, loss_func=Softmax_CrossEntropyLoss, lr=0.005)
    accuracy = validate(model, test_set)
    print('Accuracy: {}'.format(accuracy))
    model.dump_model('task2_model_SGD.pkl')


if __name__ == '__main__':
    main()
