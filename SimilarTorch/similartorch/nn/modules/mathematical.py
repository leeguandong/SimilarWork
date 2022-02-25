import numpy as np
from similartorch.autograd import Autograd


class Add(Autograd):
    def forward(self, ctx, x, y):
        return x + y

    def backward(self, ctx, grad):
        return grad, grad


class Subtract(Autograd):
    def forward(self, ctx, x, y):
        return x - y

    def backward(self, ctx, grad):
        return grad, -grad


class MatMul(Autograd):
    def forward(self, ctx, x, y):
        ctx.save_for_back(x, y)
        return x @ y

    def backward(self, ctx, grad: np.array):
        t1, t2 = ctx.data_for_back

        grad1 = grad @ np.swapaxes(t2, -1, -2)
        grad2 = np.swapaxes(t1, -1, -2) @ grad

        return grad1, grad2


class Multiply(Autograd):
    def forward(self, ctx, x, y):
        ctx.save_for_back(x, y)
        return x * y

    def backward(self, ctx, grad: np.array):
        t1, t2 = ctx.data_for_back
        return grad * t2, grad * t1


class Assign(Autograd):
    def forward(self, ctx, x):
        return x

    def backward(self, ctx, grad):
        return None


class Divide(Autograd):
    def forward(self, ctx, x, y):
        ctx.save_for_back(x, y)
        return x / y

    def backward(self, ctx, grad):
        t1, t2 = ctx.data_for_back
        grad1 = grad / t2
        grad2 = -grad1 * (t1 / t2)
        return grad1, grad2


class Negative(Autograd):
    def forward(self, ctx, x):
        return -x

    def backward(self, ctx, grad):
        return -grad


class Positive(Autograd):
    def forward(self, ctx, x):
        return np.positive(x)

    def backward(self, ctx, grad):
        return np.positive(grad)


class Power(Autograd):
    def forward(self, ctx, x, y):
        ctx.save_for_back(x, y)
        return x ** y

    def backward(self, ctx, grad):
        t1, t2 = ctx.data_for_back
        grad1 = grad * t2 * (t1 ** np.where(t2, (t2 - 1), 1))
        grad2 = grad * (t1 ** t2) * np.log(np.where(t1, t1, 1))
        return grad1, grad2


# --------------------------------------------------------------------------------
class Exp(Autograd):
    def forward(self, ctx, x):
        ctx.save_for_back(x)
        return np.exp(x)

    def backward(self, ctx, grad):
        t1, _ = ctx.data_for_back
        return grad * np.exp(t1)


class Log(Autograd):
    def forward(self, ctx, x):
        return np.log(x)

    def backward(self, ctx, grad):
        t1, _ = ctx.data_for_back
        return grad / t1
