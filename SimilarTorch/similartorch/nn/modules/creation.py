'''
@Time    : 2022/2/23 11:16
@Author  : leeguandon@gmail.com
'''
import numpy as np
from similartorch.tensor import Tensor


def emty(shape, dtype=np.float32, requires_grad=False):
    """ :return:  返回一个给定shape和dtype的Tensor
    """
    return Tensor(np.empty(shape, dtype), requires_grad=requires_grad)


def empty_like(other, dtype=None, requires_grad=False):
    """ :return: 返回一个与目标形状和类型一致的Tensor
    """
    if isinstance(other, Tensor):
        other = other.data
    return Tensor(np.empty_like(other, dtype), requires_grad=requires_grad)


def ones(shape, dtype=np.float32, requires_grad=False):
    return Tensor(np.ones(shape, dtype), requires_grad=requires_grad)


def ones_like(other, dtype=None, requires_grad=False):
    if isinstance(other, Tensor):
        other = other.data
    return Tensor(np.ones_like(other, dtype), requires_grad=requires_grad)


def zeros(shape, dtype=np.float32, requires_grad=False):
    return Tensor(np.zeros(shape, dtype), requires_grad=requires_grad)


def zeros_like(other, dtype=None, requires_grad=False):
    if isinstance(other, Tensor):
        other = Tensor.data
    return Tensor(np.zeros_like(other, dtype), requires_grad=requires_grad)


def rands(shape, requires_grad=False):
    return Tensor(np.random.normal(*shape), requires_grad=requires_grad)
