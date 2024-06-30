from ._abc import Optimizer
import numpy as np


class AdaGrad(Optimizer):
    def __init__(
        self,
        learning_rate=0.01,
    ) -> None:
        super().__init__(learning_rate)
        self.h = None

    def update(self, params, grads) -> None:
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
