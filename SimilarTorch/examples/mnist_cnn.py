'''
@Time    : 2022/2/25 17:07
@Author  : leeguandon@gmail.com
'''
import time
import numpy as np

import similartorch.nn as nn
import similartorch.optim as optim
from similartorch import CrossEntropyLoss
from similartorch.utils import Dataloader, MNIST
from similartorch.tensor import Tensor

batch_size = 64
iterations = 10
learning_rate = 0.0002

mnist_model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(800, 500),
    nn.ReLU(),
    nn.Linear(500, 10),
    nn.Softmax()
)

train_dataset = MNIST(images_path="../data/train-images-idx3-ubyte",
                      labels_path="../data/train-labels-idx1-ubyte",
                      flatten_input=False,
                      one_hot_output=True,
                      input_normalization=(0.1307, 0.3081))

test_dataset = MNIST(images_path="../data/t10k-images-idx3-ubyte",
                     labels_path="../data/t10k-labels-idx1-ubyte",
                     flatten_input=False,
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
    for it in range(iterations):
        if finished:
            break
        train_batches.shuffle()
        start = time.time()
        for i_b, (batch_in, batch_out) in enumerate(train_batches):
            model_input = Tensor(batch_in)
            good_output = Tensor(batch_out)
            model_output = mnist_model(model_input)

            err = loss(good_output, model_output)

            optimizer.zero_grad()
            err.backward()
            optimizer.step()

            if i_b % 100 == 0:
                print(i_b)

        print("epoch time: {:.2f} seconds".format(time.time() - start))
        acc = test_model_acc()
        print("model accuracy: {}".format(acc))
        if acc > 0.99:
            finished = True
            break
