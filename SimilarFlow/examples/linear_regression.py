'''
@Time    : 2022/2/18 16:42
@Author  : leeguandon@gmail.com
'''
import numpy as np
import matplotlib.pylab as plt
import similarflow as sf

input_x = np.linspace(-1, 1, 100)
input_y = input_x * 3 + np.random.randn(input_x.shape[0]) * 0.5

x = sf.Placeholder()
y = sf.Placeholder()
w = sf.Variable([[1.0]])
b = sf.Variable(0.0)

# linear = sf.add(sf.matmul(x, w), b)
linear = x * w + b

loss = sf.reduce_sum(sf.square(sf.add(linear, sf.negative(y))))
# loss = sf.reduce_sum(sf.square(linear - y))

train_op = sf.train.GradientDescentOptimizer(learning_rate=0.005).minimize(loss)

feed_dict = {x: np.reshape(input_x, (-1, 1)), y: np.reshape(input_y, (-1, 1))}
# feed_dict = {x: input_x, y: input_y}

with sf.Session() as sess:
    for step in range(20):
        # 前向
        loss_value = sess.run(loss, feed_dict)
        mse = loss_value / len(input_x)
        print(f"step:{step},loss:{loss_value},mse:{mse}")
        # 反向传播
        sess.run(train_op, feed_dict)
    w_value = sess.run(w, feed_dict=feed_dict)
    b_value = sess.run(b, feed_dict=feed_dict)
    print(f"w:{w_value},b:{b_value}")

w_value = float(w_value)
max_x, min_x = np.max(input_x), np.min(input_x)
max_y, min_y = w_value * max_x + b_value, w_value * min_x + b_value

plt.plot([max_x, min_x], [max_y, min_y], color='r')
plt.scatter(input_x, input_y)
plt.show()
