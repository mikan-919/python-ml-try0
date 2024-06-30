from numpy import ndarray
from ._abc import NetworkLayer
from ..common import cross_entropy_error, softmax


class Softmax_with_loss_Layer(NetworkLayer):
    def __init__(self) -> None:
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x: ndarray, t: ndarray) -> ndarray:
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout) -> ndarray:
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx
