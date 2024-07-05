from collections import OrderedDict
import numpy as np
from .Layer import *


class TwoLayerNet:
    def __init__(
        self, input_size, hidden_size, output_size, weight_init_std=0.01
    ) -> None:
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["B1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["B2"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Affine1"] = AffineLayer(self.params["W1"], self.params["B1"])
        self.layers["Relu1"] = ReluLayer()
        self.layers["Affine2"] = AffineLayer(self.params["W2"], self.params["B2"])
        self.lastLayer = Softmax_with_loss_Layer()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads["W1"] = self.layers["Affine1"].dW
        grads["B1"] = self.layers["Affine1"].dB
        grads["W2"] = self.layers["Affine2"].dW
        grads["B2"] = self.layers["Affine2"].dB

        return grads
