import math

import similartorch
from .module import Module
from . import init


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = similartorch.rands((0, 0.05, (in_features, out_features)), requires_grad=True)
        self.bias = similartorch.zeros([out_features], requires_grad=True) if bias else None

        self.register_parameter(("weight", self.weight), ("bias", self.bias))

        # self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return (input @ self.weight) + self.bias
