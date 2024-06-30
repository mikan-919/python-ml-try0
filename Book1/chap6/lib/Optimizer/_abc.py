from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def __init__(self, learning_rate=0.01) -> None:
        super().__init__()
        self.lr = learning_rate

    @abstractmethod
    def update(self, params, grads) -> None:
        pass
