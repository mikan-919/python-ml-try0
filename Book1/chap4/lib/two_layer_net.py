import numpy as np
from . import common as com


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params["w1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = weight_init_std * np.zeros(hidden_size)
        self.params["w2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = weight_init_std * np.zeros(output_size)

    def predict(self, x):
        a1 = np.dot(x, self.params["w1"]) + self.params["b1"]
        z1 = com.sigmoid(a1)
        a2 = np.dot(z1, self.params["w2"]) + self.params["b2"]
        y = com.softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return com.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        grads = {}

        grads["w1"] = com.numerical_gradient(lambda: self.loss(x, t), self.params["w1"])
        grads["b1"] = com.numerical_gradient(lambda: self.loss(x, t), self.params["b1"])
        grads["w2"] = com.numerical_gradient(lambda: self.loss(x, t), self.params["w2"])
        grads["b2"] = com.numerical_gradient(lambda: self.loss(x, t), self.params["b2"])

        return grads

    def gradient(self, x, t):
        """5章で学ぶ関数。誤差逆伝播法の実装"""
        W1, W2 = self.params["w1"], self.params["w2"]
        b1, b2 = self.params["b1"], self.params["b2"]
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = com.sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = com.softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads["w2"] = np.dot(z1.T, dy)
        grads["b2"] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = com.sigmoid_grad(a1) * dz1
        grads["w1"] = np.dot(x.T, da1)
        grads["b1"] = np.sum(da1, axis=0)

        return grads
