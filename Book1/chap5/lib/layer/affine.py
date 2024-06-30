import numpy as np
from numpy import ndarray
from . import _abc


class AffineLayer(_abc.NetworkLayer):
    def __init__(self, W, B) -> None:
        self.W = W
        self.B = B
        self.x = None
        self.dW = None
        self.dB = None

    def forward(self, x: ndarray) -> ndarray:
        self.x = x
        out = np.dot(x, self.W) + self.B
        return out

    def backward(self, dout: ndarray) -> ndarray:
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.dB = np.sum(dout, axis=0)
        return dx
