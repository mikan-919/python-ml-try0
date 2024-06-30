from . import _abc


class AddLayer(_abc.NetworkLayer):
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout
