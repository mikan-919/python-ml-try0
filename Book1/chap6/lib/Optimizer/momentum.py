from ._abc import Optimizer
import numpy as np


class Momentum(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v = None

    def update(self, params, grads) -> None:
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
