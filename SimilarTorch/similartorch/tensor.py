import numpy as np
from typing import Type

from similartorch.nn.modules.mathematical import Add, Subtract, Multiply, Divide, Power, Positive, Negative, MatMul
from similartorch.nn.modules.manipulation import SwapAxes, Reshape
from .autograd import Autograd


class Tensor(object):
    def __init__(self, data: np.array, requires_grad=False):
        self.data = data
        self.requires_grad = requires_grad
        self.grad = None

        if requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float32)

        self.backward_function = None
        self.backward_tensor = []
        # self.shape = self.data.shape

    def backward(self, grad=np.array([1])):
        if self.requires_grad:
            self.grad = grad + self.grad
            sum_ax = tuple(range(len(self.grad.shape) - len(self.data.shape)))
            self.grad = np.sum(self.grad, sum_ax)

        if self.backward_function is not None:
            accumulated = self.backward_function(grad)
            if len(self.backward_tensor) == 1:
                accumulated = accumulated,
            for bv, ac in zip(self.backward_tensor, accumulated):
                bv.backward(ac)

    @classmethod
    def _op(cls, Op: Type[Autograd], *input_vars):
        f = Op()
        return f(*input_vars)

    def __str__(self):
        return "<Tensor>\n" + self.data.__str__()

    def __add__(self, other):
        # from .nn import Add
        return self._op(Add, self, other)

    def __radd__(self, other):
        return self._op(Add, other, self)

    def __sub__(self, other):
        return self._op(Subtract, self, other)

    def __rsub__(self, other):
        return self._op(Subtract, other, self)

    def __matmul__(self, other):
        return self._op(MatMul, self, other)

    def __rmatmul__(self, other):
        return self._op(MatMul, other, self)

    def __mul__(self, other):
        return self._op(Multiply, self, other)

    def __rmul__(self, other):
        return self._op(Multiply, other, self)

    def __copy__(self):
        """复制当前Tensor的grad,data,requires_grad,如果当前的Tensor没有梯度，则梯度为None
        :return:
        """
        copy = Tensor(np.copy(self.data), requires_grad=self.requires_grad)
        try:
            copy.grad[:] = self.grad[:]
        except:
            pass
        return copy

    def copy(self):
        return self.__copy__()

    def numpy(self):
        return self.data.copy()

    def __len__(self):
        return len(self.data)

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def shape(self):
        return self.data.shape

    @property
    def T(self):
        pass

    def swapaxes(self, axis1, axis2):
        return SwapAxes(axis1, axis2)(self)

    def reshape(self, *new_shape):
        return Reshape(*new_shape)(self)
