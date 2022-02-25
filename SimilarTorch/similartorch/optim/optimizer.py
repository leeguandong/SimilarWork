'''
@Time    : 2022/2/25 9:51
@Author  : leeguandon@gmail.com
'''
from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, param_list: list):
        self.param_list = param_list
        self.state = {}

    def zero_grad(self):
        for param in self.param_list:
            param.grad.fill(0)

    @abstractmethod
    def step(self):
        raise NotImplementedError
