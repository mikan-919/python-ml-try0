from numpy import ndarray
from . import _abc


class ReluLayer(_abc.NetworkLayer):
    def __init__(self):
        self.mask = None

    def forward(self, x: ndarray) -> ndarray:
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout: ndarray) -> ndarray:
        dout[self.mask] = 0
        dx = dout
        return dx
