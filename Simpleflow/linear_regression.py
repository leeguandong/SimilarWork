import numpy as np
import matplotlib.pyplot as plt
import Simpleflow.simpleflow as sf

input_x = np.linspace(-1, 1, 100)
input_y = input_x * 3 + np.random.randn(input_x.shape[0]) * 0.5

# Placeholder for training data
x = sf.Placeholder()
y_ = sf.Placeholder()

# weights
w = sf.Variable([1.0], name='weight')
# Threshold
b = sf.Variable(0.0, name='threshold')

# Predicted class by model
y = x * w + b

# Build loss
loss = sf.reduce_sum(sf.square(y - y_))

# Optimizer
train_op = sf.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

# Training
# feed_dict = {x: np.reshape(input_x, (-1, 1)), y_: np.reshape(input_y, (-1, 1))}
feed_dict = {x: input_x, y_: input_y}

with sf.Session() as sess:
    for step in range(200):
        loss_value = sess.run(loss, feed_dict=feed_dict)
        mse = loss_value / len(input_x)

        if step % 1 == 0:
            print('step: {},loss:{},mse:{}'.format(step, loss_value, mse))
        sess.run(train_op, feed_dict)
    w_value = sess.run(w, feed_dict=feed_dict)
    b_value = sess.run(b, feed_dict=feed_dict)
    print('w:{},n:{}'.format(w_value, b_value))

w_value = float(w_value)
max_x, min_x = np.max(input_x), np.min(input_x)
print(max_x, min_x)
max_y, min_y = w_value * max_x + b_value, w_value * min_x + b_value
print(max_y, min_y)

# x 的输入，然后根据框架找出了 y 的预测，然后拿到两个点作图，回归的还是挺好的
plt.plot([max_x, min_x], [max_y, min_y], color='r')
plt.scatter(input_x, input_y)
plt.show()
