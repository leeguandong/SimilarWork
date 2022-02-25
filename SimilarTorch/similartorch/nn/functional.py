from similartorch.autograd import Context
from .modules import (Conv2d, Linear,
                      ReLU, Sigmoid, Softmax, Softplus, Softsign, Tanh, ArcTan,

                      )

ctx = Context()


def conv2d(input, in_channels, out_channels, kernel_size,
           stride, padding=0, add_bias=True):
    return Conv2d(in_channels, out_channels, kernel_size,
                  stride, padding=padding, add_bias=add_bias).forward(input)


def linear(input, in_features, out_features, bias=True):
    return Linear(in_features, out_features, bias=bias).forward(input)


def sigmoid(x):
    return Sigmoid().forward(ctx, x)
