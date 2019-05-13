'''
Optimizer classes for parameters optimization
'''
from Simpleflow.simpleflow.operations import Operation, compute_gradients


class GradientDescentOptimizer(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def minimize(self, loss):
        learning_rate = self.learning_rate

        class MinimizationOperation(Operation):
            def compute_output(self):
                # Get gradient table
                grad_table = compute_gradients(loss)

                # Iterate all trainable variables in graph
                for var in DEFAULT_GRAPH.trainable_variables:
                    if var in grad_table:
                        grad = grad_table[var]

                    # Update its output value
                    var.output_value -= learning_rate * grad

        return MinimizationOperation()
