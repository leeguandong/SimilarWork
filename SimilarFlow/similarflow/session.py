'''
@Time    : 2022/2/16 14:58
@Author  : leeguandon@gmail.com
'''
import numpy as np
from .graph import Operation, Placeholder, Variable, Constant


class Session(object):
    """ feedforward
    """

    def __init__(self):
        self.graph = _default_graph

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.close()

    def close(self):
        all_nodes = (self.graph.operations + self.graph.variables +
                     self.graph.constants + self.graph.placeholders)
        for node in all_nodes:
            node.output = None

    def run(self, operation, feed_dict=None):
        """   计算节点的输出值
        :param operation:
        :param feed_dict:
        :return:
        """
        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif (type(node) == Variable) or (type(node) == Constant):
                node.output = node.value
            else:  # Operation
                # 取出每个节点的值
                node.inputs = [input_node.output for input_node in node.input_nodes]
                # 拆包，调用operation的compute计算前向值
                node.output = node.compute(*node.inputs)

            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output


def traverse_postorder(operation):
    """
    通过后序遍历获取一个节点所需的所有节点的输出值，递归
    :param operation:
    :return:
    """
    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder
