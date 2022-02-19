'''
@Time    : 2022/2/17 14:41
@Author  : leeguandon@gmail.com
'''
from similarflow.operations import add


class Graph(object):
    """    computational graph
    """

    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []
        self.constants = []

    def __enter__(self):
        global _default_graph
        self.graph = _default_graph
        _default_graph = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _default_graph
        _default_graph = self.graph

    def as_default(self):
        return self


class Operation(object):
    """接受一个或者更多输入节点进行简单计算
    """

    def __init__(self, *input_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = []

        # 将当前节点的引用添加到输入节点的output_nodes，这样可以在输入节点中找到当前节点
        for node in input_nodes:
            node.output_nodes.append(self)

        # 将当前节点的引用添加到图中，方便后面对图中的资源进行回收等操作
        _default_graph.operations.append(self)

    def compute(self):
        """根据输入节点的值计算当前节点的输出值
        """
        pass

    def __add__(self, other):
        return add(self, other)


class Placeholder(object):
    """没有输入节点，节点数据是通过图建立好以后通过用户传入
    """

    def __init__(self):
        self.output_nodes = []

        _default_graph.placeholders.append(self)


class Variable(object):
    """没有输入节点，节点数据在运算过程中是可变化的
    """

    def __init__(self, initial_value=None):
        self.value = initial_value
        self.output_nodes = []

        _default_graph.variables.append(self)


class Constant(object):
    """没有输入节点，节点数据在运算过程中是不可变的
    """

    def __init__(self, value=None):
        self.value = value
        self.output_nodes = []

        _default_graph.constants.append(self)
