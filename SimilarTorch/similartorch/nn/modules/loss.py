import numpy as np
from similartorch.autograd import Autograd


class MSELoss(Autograd):
    def forward(self, ctx, target, input):
        if target.shape != input.shape:
            raise ValueError("wrong shape")

        ctx.save_for_back(target, input)
        return ((target - input) ** 2).mean()

    def backward(self, ctx, grad):
        target, input = ctx.data_for_back
        batch = target.shape[0]
        grad1 = grad * 2 * (target - input) / batch
        grad2 = grad * 2 * (input - target) / batch
        return grad1, grad2


class CrossEntropyLoss(Autograd):
    def forward(self, ctx, target, input):
        ctx.save_for_back(target, input)
        input = np.clip(input, 1e-15, 1 - 1e-15)
        return -target * np.log(input) - (1 - target) * np.log(1 - input)

    def backward(self, ctx, grad):
        target, input = ctx.data_for_back
        batch = target.shape[0]

        input = np.clip(input, 1e-15, 1 - 1e-15)
        grad1 = grad * (np.log(1 - input) - np.log(input)) / batch
        grad2 = grad * (- target / input + (1 - target) / (1 - input)) / batch
        return grad1, grad2

