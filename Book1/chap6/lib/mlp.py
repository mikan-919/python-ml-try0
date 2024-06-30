import sys, os

sys.path.append(os.pardir)
from collections import OrderedDict
import numpy as np
from . import Layer as Layer


class MLP:
    def __init__(
        self,
        input_size,
        hidden_list,
        output_size,
        weight_scale=0.01,
        init_method="he",
        activation_method="relu",
        weight_decay_lambda=0.0,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_num = len(hidden_list)
        self.weight_decay_lambda = weight_decay_lambda
        self._create_params(hidden_list, weight_scale, init_method)
        self._create_layers(hidden_list, activation_method)

    def _create_params(self, hidden_list, weight_scale=0.01, init_method="he"):
        self.params = {}
        all_size_list = [self.input_size] + hidden_list + [self.output_size]

        for i in range(1, len(all_size_list)):
            scale = weight_scale
            match init_method:
                case "he":
                    scale = np.sqrt(2.0 / all_size_list[i - 1])
                case "xa":
                    scale = np.sqrt(1.0 / all_size_list[i - 1])
                case _:
                    pass
            self.params["W" + str(i)] = scale * np.random.randn(
                all_size_list[i - 1], all_size_list[i]
            )
            self.params["b" + str(i)] = scale * np.random.randn(all_size_list[i])

    def _create_layers(self, hidden_list, activation_method="relu"):
        self.layers = OrderedDict()
        activation_layer = {"sigmoid": Layer.SigmoidLayer, "relu": Layer.ReluLayer}
        for i in range(1, len(hidden_list) + 1):
            print(i)
            self.layers["Affine" + str(i)] = Layer.AffineLayer(
                self.params["W" + str(i)], self.params["b" + str(i)]
            )
            self.layers["Activation" + str(i)] = activation_layer[activation_method]()

        i = len(hidden_list) + 1
        self.layers["Affine" + str(i)] = Layer.AffineLayer(
            self.params["W" + str(i)], self.params["b" + str(i)]
        )
        self.last_layer = Layer.Softmax_with_loss_Layer()

        pass

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params["W" + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads["W" + str(idx)] = (
                self.layers["Affine" + str(idx)].dW
                + self.weight_decay_lambda * self.layers["Affine" + str(idx)].W
            )
            grads["b" + str(idx)] = self.layers["Affine" + str(idx)].dB

        return grads
