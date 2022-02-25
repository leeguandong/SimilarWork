'''
@Time    : 2022/2/25 15:47
@Author  : leeguandon@gmail.com
'''
import numpy as np
import similartorch
import similartorch.nn as nn
import similartorch.optim as optim
from similartorch.tensor import Tensor
from similartorch import CrossEntropyLoss
from similartorch.utils import MNIST, Dataloader

batch_size = 64
epoches = 10
learning_rate = 0.0001

mnist_model = nn.Sequential(
    nn.Linear(28 * 28, 300),
    nn.ReLU(),
    nn.Linear(300, 300),
    nn.ReLU(),
    nn.Linear(300, 10),
    nn.Softmax()
)

# class mnist_model():
#     def __init__(self):
#         pass
#
#     def forward(self):
#         pass

train_dataset = MNIST(images_path="../data/train-images-idx3-ubyte",
                      labels_path="../data/train-labels-idx1-ubyte",
                      flatten_input=True,
                      one_hot_output=True,
                      input_normalization=(0.1307, 0.3081))

test_dataset = MNIST(images_path="../data/t10k-images-idx3-ubyte",
                     labels_path="../data/t10k-labels-idx1-ubyte",
                     flatten_input=True,
                     one_hot_output=True,
                     input_normalization=(0.1307, 0.3081))

train_dataloader = Dataloader(train_dataset)
test_datasloader = Dataloader(test_dataset)

train_batches = train_dataloader.get_batch_iterable(batch_size)
test_batches = test_datasloader.get_batch_iterable(batch_size)

optimizer = optim.Adam(mnist_model.parameters(), learning_rate)
loss = CrossEntropyLoss()


def test_model_acc():
    correct = 0
    for test_batch_in, test_batch_out in test_batches:
        test_output = mnist_model(Tensor(test_batch_in)).data
        correct += np.sum(np.argmax(test_output, axis=1) == np.argmax(test_batch_out, axis=1))

    my_acc = correct / len(test_dataset)
    return my_acc


if __name__ == "__main__":
    finished = False
    for it in range(epoches):
        if finished:
            break
        train_batches.shuffle()

        for i_b, (batch_in, batch_out) in enumerate(train_batches):
            model_input = Tensor(batch_in)

            good_output = Tensor(batch_out)
            model_output = mnist_model(model_input)

            err = loss(good_output, model_output)
            optimizer.zero_grad()
            err.backward()

            optimizer.step()

        acc = test_model_acc()
        print("model accuracy: {}".format(acc))
        if acc > 0.97:
            finished = True
            break
