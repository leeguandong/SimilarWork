import math
import numpy as np
import similartorch
from similartorch.tensor import Tensor
from .img2col import Img2Col
from .module import Module
from . import init


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, add_bias=True):
        super(Conv2d, self).__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.add_bias = add_bias

        self.weight = similartorch.rands([0, 0.05, (self.out_channels, self.in_channels,
                                                    self.kernel_size[0], self.kernel_size[1])], requires_grad=True)
        if add_bias:
            self.bias = similartorch.zeros(out_channels, np.float32, requires_grad=True)
            self.register_parameter(("weight", self.weight), ("bias", self.bias))
        else:
            self.register_parameter(("weight", self.weight))

        self.img2col = Img2Col(self.kernel_size, self.stride)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        img2col = self.img2col(input)
        output = self.weight.reshape(self.weight.shape[0], -1) @ img2col

        img_w = input.shape[-1]
        img_h = input.shape[-2]
        new_w = (img_w - self.kernel_size[0]) // self.stride[0] + 1
        new_h = (img_h - self.kernel_size[1]) // self.stride[1] + 1

        batch_input = len(input.shape) == 4
        if batch_input:
            output_shape = (input.shape[0], self.out_channels, new_h, new_w)
        else:
            output_shape = (self.out_channels, new_h, new_w)

        if self.add_bias:
            output = (output.swapaxes(-1, -2) + self.bias).swapaxes(-1, -2)

        return output.reshape(*output_shape)
