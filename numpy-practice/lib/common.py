import numpy as np
from rich.progress import track


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def numerical_gradient(f, x, title="勾配の計算"):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in track(range(x.size), description=title):
        tmp_val = x.flat[idx]

        x.flat[idx] = tmp_val + h
        fxh1 = f(x)

        x.flat[idx] = tmp_val - h
        fxh2 = f(x)

        grad.flat[idx] = (fxh1 - fxh2) / (2 * h)
        x.flat[idx] = tmp_val

    return grad
