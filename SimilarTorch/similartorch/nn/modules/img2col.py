'''
@Time    : 2022/2/22 14:00
@Author  : leeguandon@gmail.com
'''
import numpy as np

from typing import Tuple, Union
from similartorch.autograd import Autograd, Context


class Img2Col(Autograd):
    """
    卷积的原理，conv
    """

    def __init__(self, kernel_size, stride: Union[int, Tuple[int, int]] = 1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        self.kernel_size = kernel_size
        self.stride = stride

    @staticmethod
    def img2col_forward(kernel_size, stride, merge_channels, image):
        has_batches = len(image.shape) == 4

        img_w = image.shape[-1]
        img_h = image.shape[-2]
        channels = image.shape[-3]

        new_w = (img_w - kernel_size[0]) // stride[0] + 1
        new_h = (img_h - kernel_size[1]) // stride[1] + 1

        if merge_channels:
            ret_shape = (channels * kernel_size[0] * kernel_size[1], new_w * new_h)
            flattened_part_shape = (-1,)
        else:
            ret_shape = (channels, kernel_size[0] * kernel_size[1], new_w * new_h)
            flattened_part_shape = (channels, -1)

        if has_batches:
            ret_shape = (image.shape[0], *ret_shape)
            flattened_part_shape = (image.shape[0], *flattened_part_shape)

        ret_image = np.zeros(ret_shape)
        for i in range(new_h):
            for j in range(new_w):
                part = image[
                       ...,
                       i * stride[1]:i * stride[1] + kernel_size[1],
                       j * stride[0]:j * stride[0] + kernel_size[0]
                       ]
                part = np.reshape(part, flattened_part_shape)
                ret_image[..., :, i * new_w + j] = part
        return ret_image

    @staticmethod
    def img2col_backward(kernel_size, stride, back_shape, grad):
        channels = back_shape[-3]
        img_w = back_shape[-1]

        back_w = (img_w - kernel_size[0]) // stride[0] + 1
        ret_grad = np.zeros(back_shape, dtype=np.float32)

        for i in range(grad.shape[-1]):
            col = grad[..., :, i]
            col = np.reshape(col, (-1, channels, kernel_size[1], kernel_size[0]))

            h_start = (i // back_w) * stride[1]
            w_start = (i % back_w) * stride[0]

            ret_grad[..., h_start:h_start + kernel_size[1], w_start:w_start + kernel_size[0]] += col
        return ret_grad

    def forward(self, ctx: Context, image: np.array):
        """
        :param ctx:
        :param image: [N,C,H,W],[C,H,W]
        :return:
        """
        ctx.save_for_back(image.shape)
        return self.img2col_forward(
            self.kernel_size,
            self.stride,
            True,
            image
        )

    def backward(self, ctx: Context, grad: np.array = None):
        back_shape, = ctx.data_for_back

        return self.img2col_backward(
            self.kernel_size,
            self.stride,
            back_shape,
            grad
        )
