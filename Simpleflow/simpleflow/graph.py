'''
将定义好的节点放到一个图中统一保管，定义 graph 类来存放创建的节点，方便统一操作图中节点的资源
模仿 tensorflow，给 graph 添加上下文管理器协议方法使其成为一个上下文管理器，同时添加一个 as_default 方法

这样在进入 with 代码块之前先保存旧的默认图对象然后将当前图赋值给全局图对象，这样 with 代码块中的节点默认会添加到
当前的图中。最后退出 with 代码块时再对图进行恢复即可。这样我们可以按照 TensorFlow 的方式来在某个图中创建节点.
'''


# Computational graph definition
class Graph(object):
    def __init__(self):
        self.operations, self.constants, self.placeholders = [], [], []
        self.variables, self.trainable_variables = [], []

    def __enter__(self):
        # reset default graph
        global DEFAULT_GRAPH
        self.old_graph = DEFAULT_GRAPH
        DEFAULT_GRAPH = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # recover default graph
        global DEFAULT_GRAPH
        DEFAULT_GRAPH = self.old_graph

    def as_default(self):
        # set this graph as global default graph
        return self
