'''
@Time    : 2022/2/25 9:52
@Author  : leeguandon@gmail.com
'''
import numpy as np

from similartorch.autograd import Autograd


class Flatten(Autograd):
    def forward(self, ctx, input):
        ctx.save_for_back(input.shape)
        return np.reshape(input, (input.shape[0], -1))

    def backward(self, ctx, grad):
        back_shape, = ctx.data_for_back
        return np.reshape(grad, back_shape)
