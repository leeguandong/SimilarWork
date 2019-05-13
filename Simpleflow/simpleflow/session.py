'''
把节点放到计算图中，然后通过遍历得到某一个节点的所有计算所需的节点列表，最终得到输出值
'''
from Simpleflow.simpleflow.operations import Operation, Placeholder


class Session(object):
    def __init__(self):
        self.graph = DEFAULT_GRAPH

    def __enter__(self):
        # Context management protocal method called before `with-block`.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Context management protocal method called before `with-block`.
        self.close()

    def close(self):
        # free all output values in nodes
        all_nodes = (self.graph.constants + self.graph.variables + self.graph.placeholders +
                     self.graph.operations + self.graph.trainable_variables)
        for node in all_nodes:
            node.output_value = None

    def run(self, operation, feed_dict=None):
        # compute the output of an operation
        postorder_nodes = _get_prerequisite(operation)

        for node in postorder_nodes:
            if type(node) is Placeholder:
                node.output_value = feed_dict[node]
            else:
                node.compute_output()
        return operation.output_value


def _get_prerequisite(operation):
    # perform a post-order traversal to get a list of nodes to be computed in order
    postorder_nodes = []

    def postorder_traverse(operation):
        if isinstance(operation, Operation):
            for input_node in operation.input_nodes:
                postorder_traverse(input_node)
        postorder_nodes.append(operation)

    postorder_traverse(operation)

    return postorder_nodes
