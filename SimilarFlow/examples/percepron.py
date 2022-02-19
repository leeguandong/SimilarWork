import numpy as np
import similarflow as sf
import matplotlib.pyplot as plt

# Create red points centered at (-2, -2)
red_points = np.random.randn(50, 2) - 2 * np.ones((50, 2))

# Create blue points centered at (2, 2)
blue_points = np.random.randn(50, 2) + 2 * np.ones((50, 2))

X = sf.Placeholder()
y = sf.Placeholder()
W = sf.Variable(np.random.randn(2, 2))
b = sf.Variable(np.random.randn(2))

p = sf.softmax(sf.add(sf.matmul(X, W), b))

loss = sf.negative(sf.reduce_sum(sf.reduce_sum(sf.multiply(y, sf.log(p)), axis=1)))

train_op = sf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

feed_dict = {
    X: np.concatenate((blue_points, red_points)),
    y: [[1, 0]] * len(blue_points) + [[0, 1]] * len(red_points)
}

with sf.Session() as sess:
    for step in range(100):
        loss_value = sess.run(loss, feed_dict)
        if step % 10 == 0:
            print(f"step:{step},loss:{loss_value}")
        sess.run(train_op, feed_dict)

    # Print final result
    W_value = sess.run(W)
    print("Weight matrix:\n", W_value)
    b_value = sess.run(b)
    print("Bias:\n", b_value)

# Plot a line y = -x
x_axis = np.linspace(-4, 4, 100)
y_axis = -W_value[0][0] / W_value[1][0] * x_axis - b_value[0] / W_value[1][0]
plt.plot(x_axis, y_axis)

# Add the red and blue points
plt.scatter(red_points[:, 0], red_points[:, 1], color='red')
plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue')
plt.show()
