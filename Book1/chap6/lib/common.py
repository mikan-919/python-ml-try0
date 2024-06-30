import numpy as np


def identity_function(x):
    return x


def sigmoid(x):
    sigmoid_range = 34.538776394910684
    x = np.clip(x, -sigmoid_range, sigmoid_range)

    # シグモイド関数
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    sig_x = sigmoid(x)
    return (1.0 - sig_x) * sig_x


def softmax(x):
    c = np.max(x, axis=-1, keepdims=True)
    exp_a = np.exp(x - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a, axis=-1, keepdims=True)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
