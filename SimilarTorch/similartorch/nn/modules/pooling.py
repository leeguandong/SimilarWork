import numpy as np

from abc import ABC
from similartorch.autograd import Autograd, Context
from .img2col import Img2Col


class BasePool(Autograd, ABC):
    def __init__(self, kernel_size, stride=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

    @staticmethod
    def _fill_col(to_fill, new_shape):
        repeats = new_shape[-2]
        ret = np.repeat(to_fill, repeats, -2)
        ret = np.reshape(ret, new_shape)
        return ret


class MaxPool2d(BasePool):
    def forward(self, ctx: Context, input):
        img_w = input.shape[-1]
        img_h = input.shape[-2]
        channels = input.shape[-3]

        new_w = (img_w - self.kernel_size[0]) // self.stride[0] + 1
        new_h = (img_h - self.kernel_size[1]) // self.stride[1] + 1

        img_out = Img2Col.img2col_forward(self.kernel_size, self.stride, False, input)
        maxed = np.max(img_out, -2)

        ctx.save_for_back(img_out, input.shape, maxed.shape)
        return np.reshape(maxed, (-1, channels, new_h, new_w))

    def backward(self, ctx: Context, grad: np.array = None):
        """切成一小块算max，计算完毕之后再把shape转回去
        """
        reshaped_image, back_shape, maxed_shape = ctx.data_for_back

        grad = np.reshape(grad, maxed_shape)
        mask = (reshaped_image == np.max(reshaped_image, -2, keepdims=True))
        new_grad = self._fill_col(grad, reshaped_image.shape)

        new_grad = np.where(mask, new_grad, 0)
        return Img2Col.img2col_backward(self.kernel_size, self.stride, back_shape, new_grad)


class AvgPool2d(BasePool):
    def forward(self, ctx: Context, input):
        img_w = input.shape[-1]
        img_h = input.shape[-2]
        channels = input.shape[-3]

        new_w = (img_w - self.kernel_size[0]) // self.stride[0] + 1
        new_h = (img_h - self.kernel_size[1]) // self.stride[1] + 1

        img_out = Img2Col.img2col_forward(self.kernel_size, self.stride, False, input)
        averaged = np.average(img_out, -2)
        ctx.save_for_back(img_out, input.shape, averaged.shape)
        return np.reshape(averaged, (-1, channels, new_h, new_w))

    def backward(self, ctx, grad):
        reshaped_image, back_shape, averaged_shape = ctx.data_for_back

        grad = np.reshape(grad, averaged_shape)
        new_grad = self._fill_col(grad, reshaped_image.shape) / (self.kernel_size[0] * self.kernel_size[1])

        return Img2Col.img2col_backward(self.kernel_size, self.stride, back_shape, new_grad)
