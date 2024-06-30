from abc import ABC, abstractmethod
import numpy as np


class NetworkLayer(ABC):
    @abstractmethod
    def forward(self, *args: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dout: np.ndarray) -> np.ndarray:
        pass
