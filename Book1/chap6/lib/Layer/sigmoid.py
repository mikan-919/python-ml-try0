from numpy import ndarray
from . import _abc
from .. import common


class SigmoidLayer(_abc.NetworkLayer):
    def __init__(self) -> None:
        self.out = None

    def forward(self, x: ndarray) -> ndarray:
        out = common.sigmoid(x)
        self.out = out
        return out

    def backward(self, dout: ndarray) -> ndarray:
        dx = dout * (1.0 - self.out) * self.out
        return dx
