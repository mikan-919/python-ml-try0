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
        z1 = com.sigmoid(np.dot(x, self.params["w1"]) + self.params["b1"])
        y = com.softmax(np.dot(z1, self.params["w2"]) + self.params["b2"])

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
        loss_w = lambda W: self.loss(x, t)

        grads = {}

        grads["w1"] = com.numerical_gradient(loss_w, self.params["w1"])
        grads["b1"] = com.numerical_gradient(loss_w, self.params["b1"])
        grads["w2"] = com.numerical_gradient(loss_w, self.params["w2"])
        grads["b2"] = com.numerical_gradient(loss_w, self.params["b2"])

        return grads
