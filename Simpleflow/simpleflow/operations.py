'''
operation classes in computational graph
'''
from queue import Queue
import numpy as np


class Operation(object):
    '''
    input_nodes: 里面存放与当前节点相连接的输入节点的引用，这可能还涉及到深浅拷贝的问题，类似于指针
    output_nodes: 存放以当前节点作为输入的节点，也就是当前节点的去向，引用，指针
    output_value: 存储当前节点的数值，如果是 add，则是 output_nodes 之和
    name: 当前节点的名称
    graph：此节点所属的图
    compute_output: 根据输入节点的值计算当前节点的输出值
    compute_gradient: 根据操作属性和当前节点的值计算梯度
    '''

    def __init__(self, *input_nodes, name=None):
        # Nodes received by this operation.
        self.input_nodes = input_nodes

        # Nodes that receive this operation node as input.
        self.output_nodes = []

        # Output value of this operation in session execution.
        self.output_value = None

        # Add this operation node to destination lists in its input nodes.
        for node in input_nodes:
            node.output_nodes.append(self)

        # Graph the operation belongs to.
        self.graph = DEFAULT_GRAPH

        # Add this operation to default graph.
        self.graph.operations.append(self)

        self.name = name

    def compute_output(self):
        '''
        compute and return the output value of the operation
        '''
        raise NotImplementedError

    def compute_gradient(self, grad=None):
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


## x+y grad_x:1,grad_y:1    xy  grad_x:y.,grad_y:x
# so in compute_gradient function you will see this

# Add operation
class Add(Operation):
    def __init__(self, x, y, name=None):
        '''
        x,y all type of operation,variable,constant
        '''
        super(Add, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.add(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        '''
        :param grad: The gradient of other operation wrt the addition output.
        :type: grad: number or a ndarray, default value is 1.0
        '''
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        grad_wrt_x = grad
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]


def add(x, y, name=None):
    return Add(x, y, name)


# Multiplication opseration
class Multiply(Operation):
    def __init__(self, x, y, name=None):
        super(Multiply, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.multiply(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        ''' Compute and return gradients for this operation wrt input values.

        :param grad: The gradient of other operation wrt the mutiply output.
        :type grad: number or a ndarray.
        '''
        x, y = [node.output_value for node in self.input_nodes]

        if grad is None:
            grad = np.ones_like(self.output_value)

        grad_wrt_x = grad * y
        while np.ndim(grad_wrt_x) > len(np.shape(x)):
            grad_wrt_x = np.sum(grad_wrt_x, axis=0)
        for axis, size in enumerate(np.shape(x)):
            if size == 1:
                grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)

        grad_wrt_y = grad * x
        while np.ndim(grad_wrt_y) > len(np.shape(y)):
            grad_wrt_y = np.sum(grad_wrt_y, axis=0)
        for axis, size in enumerate(np.shape(y)):
            if size == 1:
                grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)

        return [grad_wrt_x, grad_wrt_y]


def multiply(x, y, name=None):
    return Multiply(x, y, name)


# Matrix multipilication operation
class MatMul(Operation):
    def __init__(self, x, y, name=None):
        super(self.__class__, self).__init__(x, y, name=name)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.dot(x.output_value, y.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        # Get input values.
        x, y = [node.output_value for node in self.input_nodes]

        # Default gradient wrt the matmul output.
        if grad is None:
            grad = np.ones_like(self.output_value)

        dfdx = np.dot(grad, np.transpose(y))
        dfdy = np.dot(np.transpose(x), grad)

        return [dfdx, dfdy]


def matmul(x, y, name=None):
    return MatMul(x, y, name)


# sigmoid operation
class Sigmoid(Operation):
    def __init__(self, x, name=None):
        super(Sigmoid, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = 1 / (1 + np.exp(-x.output_value))
        return self.output_value

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad * self.output_value * (1 - self.output_value)


def sigmoid(x, name=None):
    return Sigmoid(x, name=name)


# Logarithm operation
class Log(Operation):
    def __init__(self, x, name=None):
        super(Log, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.log(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        x = self.input_nodes[0].output_value
        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad * 1 / x


def log(x, name=None):
    return Log(x, name=name)


# Negative operation
class Negative(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = -x.output_value
        return self.output_value

    def compute_gradient(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.output_value)
        return -grad


def negative(x, name=None):
    return Negative(x, name=name)


# Reduce sum operation
class ReduceSum(Operation):
    def __init__(self, x, axis=None):
        super(ReduceSum, self).__init__(x)
        self.axis = axis

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.sum(x.output_value, self.axis)
        return self.output_value

    def compute_gradient(self, grad=None):
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)

        output_shape = np.array(np.shape(input_value))
        output_shape[self.axis] = 1.0
        tile_scaling = np.shape(input_value) // output_shape
        grad = np.reshape(grad, output_shape)
        return np.tile(grad, tile_scaling)


def reduce_sum(x, axis=None):
    return ReduceSum(x, axis=axis)


# square operation
class Square(Operation):
    def __init__(self, x, name=None):
        super(self.__class__, self).__init__(x, name=name)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.square(x.output_value)
        return self.output_value

    def compute_gradient(self, grad=None):
        input_value = self.input_nodes[0].output_value

        if grad is None:
            grad = np.ones_like(self.output_value)
        return grad * np.multiply(2.0, input_value)


def square(x, name=None):
    return Square(x, name=name)


# Constant node
class Constant(object):
    '''
    没有 input_node ,constant 自己初始化
    '''

    def __init__(self, value, name=None):
        self.value = value

        self.output_value = None

        self.output_nodes = []

        self.name = name

        DEFAULT_GRAPH.constants.append(self)

    def compute_output(self):
        if self.output_value is None:
            self.output_value = self.value
        return self.output_value

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


def constant(value, name=None):
    return Constant(value, name=name)


# Variable node
class Variable(object):
    def __init__(self, initial_value=None, name=None, trainable=True):
        # Variable initial value.
        self.initial_value = initial_value

        # Output value of this operation in session execution.
        self.output_value = None

        # Nodes that receive this variable node as input.
        self.output_nodes = []

        # Variable name.
        self.name = name

        # Graph the variable belongs to.
        self.graph = DEFAULT_GRAPH

        # Add to the currently active default graph.
        self.graph.variables.append(self)
        if trainable:
            self.graph.trainable_variables.append(self)

    def compute_output(self):
        ''' Compute and return the variable value.
        '''
        if self.output_value is None:
            self.output_value = self.initial_value
        return self.output_value

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


def variable(value, name=None):
    return Variable(value, name=name)


# Placeholder node
class Placeholder(object):
    def __init__(self, name=None):
        # Output value of this operation in session execution.
        self.output_value = None

        # Nodes that receive this placeholder node as input.
        self.output_nodes = []

        # Placeholder node name.
        self.name = name

        # Graph the placeholder node belongs to.
        self.graph = DEFAULT_GRAPH

        # Add to the currently active default graph.
        self.graph.placeholders.append(self)

    def __add__(self, other):
        return Add(self, other)

    def __neg__(self):
        return Negative(self)

    def __sub__(self, other):
        return Add(self, Negative(other))

    def __mul__(self, other):
        return Multiply(self, other)


def placeholder(name=None):
    return Placeholder(name=name)


# Function for gradients computation
def compute_gradients(target_op):
    '''
    Backpropagation implementation computing gradient of target operation wrt all the other connected nodes
    '''
    # A dict containing a mapping between node and gradient value of target_op wrt the node's output
    grad_table = {}

    # The gradient wrt target_op itself is 1
    grad_table[target_op] = np.ones_like(target_op.output_value)

    # Perform a breadth-first search staring from the target_op in graph
    queue = Queue()
    queue.put(target_op)

    # Set for visited nodes
    visited = set()
    visited.add(target_op)

    while not queue.empty():
        node = queue.get()

        # compute gradient wrt the node's output
        if node != target_op:
            grads_wrt_node_output = []

            for output_node in node.output_nodes:
                # Retrieve the gradient wrt output_node's OUTPUT
                grad_wrt_output_node_output = grad_table[output_node]

                # Compute the gradient wrt current node's output
                grad_wrt_node_output = output_node.compute_gradient(grad_wrt_output_node_output)
                if len(output_node.input_nodes) > 1:
                    input_node_index = output_node.input_nodes.index(node)
                    grads_wrt_node_output.append(grad_wrt_node_output[input_node_index])
                else:
                    grads_wrt_node_output.append(grad_wrt_node_output)

            # Sum all gradients wrt node's output
            tot_grad_wrt_node_output = sum(grads_wrt_node_output)
            grad_table[node] = tot_grad_wrt_node_output

        # put adjecent nodes to queue
        if hasattr(node, 'input_nodes'):
            for input_node in node.input_nodes:
                if input_node not in visited:
                    visited.add(input_node)
                    queue.put(input_node)

    return grad_table
