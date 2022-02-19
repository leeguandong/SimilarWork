'''
@Time    : 2022/2/17 18:58
@Author  : leeguandon@gmail.com
'''
from .operations import *

_gradient_registry = {}


class RegisterGradient(object):
    def __init__(self, op_type):
        self._op_type = eval(op_type)

    def __call__(self, f):
        _gradient_registry[self._op_type] = f
        return f


@RegisterGradient("add")
def _add_gradient(op, grad):
    """   求和矩阵求导，行相加，列相加
    :param op:
    :param grad:
    :return:
    """
    x, y = op.inputs[0], op.inputs[1]

    grad_wrt_x = grad
    while np.ndim(grad_wrt_x) > len(np.shape(x)):
        grad_wrt_x = np.sum(grad_wrt_x, axis=0)
    for axis, size in enumerate(np.shape(x)):
        if size == 1:
            grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

    grad_wrt_y = grad
    while np.ndim(grad_wrt_y) > len(np.shape(y)):
        grad_wrt_y = np.sum(grad_wrt_y, axis=0)
    for axis, size in enumerate(np.shape(y)):
        if size == 1:
            grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

    return [grad_wrt_x, grad_wrt_y]


@RegisterGradient("matmul")
def _matmul_gradient(op, grad):
    """ 求x的梯度：y的转置，求y的梯度：x的转置
    :param op:
    :param grad:
    :return:
    """
    x, y = op.inputs[0], op.inputs[1]
    return [np.dot(grad, np.transpose(y)), np.dot(np.transpose(x), grad)]


@RegisterGradient("sigmoid")
def _sigmoid_gradient(op, grad):
    sigmoid = op.output
    return grad * sigmoid * (1 - sigmoid)


@RegisterGradient("softmax")
def _softmax_gradient(op, grad):
    """ softmax 倒数
    https://stackoverflow.com/questions/40575841/numpy-calculate-the-derivative-of-the-softmax-function
    :param op:
    :param grad:
    :return:
    """
    softmax = op.output
    return (grad - np.reshape(np.sum(grad * softmax, 1), [-1, 1])) * softmax


@RegisterGradient("log")
def _log_gradient(op, grad):
    x = op.inputs[0]
    return grad / x


@RegisterGradient("multiply")
def _multiply_gradient(op, grad):
    x, y = op.inputs[0], op.inputs[1]
    return [grad * y, grad * x]


@RegisterGradient("negative")
def _negative_gradient(op, grad):
    return -grad


@RegisterGradient("square")
def _square_gradient(op, grad):
    x = op.inputs[0]
    return grad * np.multiply(2.0, x)


@RegisterGradient("reduce_sum")
def _reduce_sum_gradient(op, grad):
    x = op.inputs[0]

    output_shape = np.array(np.shape(x))
    output_shape[op.axis] = 1
    tile_scaling = np.shape(x) // output_shape
    grad = np.reshape(grad, output_shape)
    return np.tile(grad, tile_scaling)
