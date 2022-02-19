import numpy as np
import similarflow as sf
import matplotlib.pyplot as plt

# Create two clusters of red points centered at (0, 0) and (1, 1), respectively.
red_points = np.concatenate((
    0.2 * np.random.randn(25, 2) + np.array([[0, 0]] * 25),
    0.2 * np.random.randn(25, 2) + np.array([[1, 1]] * 25)
))

# Create two clusters of blue points centered at (0, 1) and (1, 0), respectively.
blue_points = np.concatenate((
    0.2 * np.random.randn(25, 2) + np.array([[0, 1]] * 25),
    0.2 * np.random.randn(25, 2) + np.array([[1, 0]] * 25)
))

# Plot them
plt.scatter(red_points[:, 0], red_points[:, 1], color='red')
plt.scatter(blue_points[:, 0], blue_points[:, 1], color='blue')
plt.show()

X = sf.Placeholder()
y = sf.Placeholder()
W_hidden = sf.Variable(np.random.randn(2, 2))
b_hidden = sf.Variable(np.random.randn(2))
p_hidden = sf.sigmoid(sf.add(sf.matmul(X, W_hidden), b_hidden))

W_output = sf.Variable(np.random.randn(2, 2))
b_output = sf.Variable(np.random.rand(2))
p_output = sf.softmax(sf.add(sf.matmul(p_hidden, W_output), b_output))

loss = sf.negative(sf.reduce_sum(sf.reduce_sum(sf.multiply(y, sf.log(p_output)), axis=1)))

train_op = sf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(loss)

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
    W_hidden_value = sess.run(W_hidden)
    print("Hidden layer weight matrix:\n", W_hidden_value)
    b_hidden_value = sess.run(b_hidden)
    print("Hidden layer bias:\n", b_hidden_value)
    W_output_value = sess.run(W_output)
    print("Output layer weight matrix:\n", W_output_value)
    b_output_value = sess.run(b_output)
    print("Output layer bias:\n", b_output_value)

# Visualize classification boundary
xs = np.linspace(-2, 2)
ys = np.linspace(-2, 2)
pred_classes = []
for x in xs:
    for y in ys:
        pred_class = sess.run(p_output, feed_dict={X: [[x, y]]})[0]
        pred_classes.append((x, y, pred_class.argmax()))
xs_p, ys_p = [], []
xs_n, ys_n = [], []
for x, y, c in pred_classes:
    if c == 0:
        xs_n.append(x)
        ys_n.append(y)
    else:
        xs_p.append(x)
        ys_p.append(y)
plt.plot(xs_p, ys_p, 'ro', xs_n, ys_n, 'bo')
plt.show()
