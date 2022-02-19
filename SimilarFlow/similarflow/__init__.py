'''
@Time    : 2022/2/16 14:58
@Author  : leeguandon@gmail.com
'''
import similarflow.train
from .graph import Operation, Variable, Graph, Placeholder, Constant
from .operations import matmul, add, multiply, sigmoid, softmax, square, reduce_sum, log, negative
from .session import Session
from .gradients import RegisterGradient

# create a default graph
import builtins

_default_graph = builtins._default_graph = Graph()

__all__ = [
    "Operation", "Variable", "Graph", "Placeholder", "Constant",
    "Session",
    "matmul", "add", "multiply", "reduce_sum", "square", "softmax", "sigmoid", "log", "negative",
    "RegisterGradient"
]
