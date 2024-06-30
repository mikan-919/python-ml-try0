from ._abc import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate=0.01) -> None:
        super().__init__(learning_rate)

    def update(self, params, grads) -> None:
        for key in params.keys():
            params[key] -= self.lr * grads[key]
