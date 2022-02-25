'''
@Time    : 2022/2/25 9:51
@Author  : leeguandon@gmail.com
'''
import numpy as np
from .optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, param_list: list, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(Adam, self).__init__(param_list)

        self.lr = learning_rate

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

    @staticmethod
    def initialize_state(state, param):
        state["step"] = 0
        state["m"] = np.zeros(param.grad.shape)
        state["v"] = np.zeros(param.grad.shape)

    def step(self):
        for param in self.param_list:
            if param.grad is None:
                continue

            if param not in self.state:
                self.state[param] = {}

            state = self.state[param]

            if len(state) == 0:
                self.initialize_state(state, param)

            state["step"] += 1
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * param.grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * param.grad

            m_hat = state["m"] / (1 - self.beta1 ** state["step"])
            v_hat = state["v"] / (1 - self.beta2 ** state["step"])
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

