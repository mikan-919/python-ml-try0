import os.path as path
import pickle
import numpy as np


def __flatten(a):
    return np.reshape(a, (-1, 784))


def load_mnist(flatten=False, normalize=False, one_hot=False):
    mnist_pickle_path = path.join(path.dirname(__file__), "mnist.pickle")
    if path.isfile(mnist_pickle_path):
        with open(mnist_pickle_path, "rb") as f:
            mnist_dataset = pickle.load(f)
    else:
        from keras._tf_keras.keras.datasets import mnist

        mnist_dataset = mnist.load_data()
        with open(mnist_pickle_path, "wb") as f:
            pickle.dump(mnist_dataset, f)

    train_i = mnist_dataset[0][0]
    train_l = mnist_dataset[0][1]
    test_i = mnist_dataset[1][0]
    test_l = mnist_dataset[1][1]

    if flatten:
        train_i = __flatten(train_i)
        train_l = train_l
        test_i = __flatten(test_i)
        test_l = test_l
    if normalize:
        train_i = train_i / 255
        test_i = test_i / 255
    if one_hot:
        train_l = np.eye(10)[train_l]
        test_l = np.eye(10)[test_l]
    return (train_i, train_l), (test_i, test_l)
