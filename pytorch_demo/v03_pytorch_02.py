"""
用numpy实现两层神经网络
"""
import numpy as np

N, D_in, H, D_out = 64, 100, 100, 10
# 随机创建一些训练数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-5

for it in range(500):
    # Forward pass
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # compute loss
    loss = np.square(y_pred - y).sum()
    print(it, loss)

    # Backward pass
    # compute the gradient
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # update weights of w1 & w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
