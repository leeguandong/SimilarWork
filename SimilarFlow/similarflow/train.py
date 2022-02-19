'''
@Time    : 2022/2/18 15:32
@Author  : leeguandon@gmail.com
'''
from queue import Queue

from .graph import Operation, Variable, Constant
from .gradients import _gradient_registry


def compute_gradients(loss):
    """ 已知每个节点中输出对输入的梯度，从后往前反向搜索与损失节点相关联的节点进行反向传播计算梯度。
    若我们需要计算其他节点关于loss的梯度需要以损失节点为启动对计算图进行广度优先搜索，在搜索过程中
    针对每个节点的梯度计算便可以一边遍历一边计算计算节点对遍历节点的梯度，可以用dict将节点与梯度进行保存。

    使用一个先进先出的队列控制遍历顺序，一个集合对象存储已访问的节点防止重复访问，然后遍历的时候计算梯度并将
    梯度放到grad_table中
    :param loss:
    :return:
    """
    grad_table = {}  # 存放节点的梯度
    grad_table[loss] = 1

    visited = set()
    queue = Queue()
    visited.add(loss)
    queue.put(loss)

    while not queue.empty():
        node = queue.get()

        # 该节点不是loss节点，先遍历进queue
        if node != loss:
            grad_table[node] = 0

            for output_node in node.output_nodes:
                lossgrad_wrt_output_node_output = grad_table[output_node]

                output_node_op_type = output_node.__class__
                bprop = _gradient_registry[output_node_op_type]

                lossgrads_wrt_output_node_inputs = bprop(output_node, lossgrad_wrt_output_node_output)

                if len(output_node.input_nodes) == 1:
                    grad_table[node] += lossgrads_wrt_output_node_inputs
                else:
                    # 若一个节点有多个输出，则多个梯度求和
                    node_index_in_output_node_inputs = output_node.input_nodes.index(node)
                    lossgrad_wrt_node = lossgrads_wrt_output_node_inputs[node_index_in_output_node_inputs]
                    grad_table[node] += lossgrad_wrt_node

        # 把节点存入到队列中
        if hasattr(node, "input_nodes"):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table


class GradientDescentOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute(self):
                grad_table = compute_gradients(loss)

                for node in grad_table:
                    if type(node) == Variable or type(node) == Constant:
                        grad = grad_table[node]
                        node.value -= learning_rate * grad

        return MinimizationOperation()
