import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1, 2, 3, 4])
y_data = np.array([4, 7, 10, 13])
w = 0  # 初始化参数
b = 0  # 初始化参数
e = 0.01  # 学习率
loss_list = []


def cal_gradient(x_data, y_data):
    gradient = []
    gradient.append(sum(2 * x_data * (x_data * w + b - y_data)) / len(x_data))
    gradient.append(np.sum(2 * (x_data * w + b - y_data)) / len(x_data))
    return gradient


for epoch in range(1, 51, 1):
    y_hat = x_data * w + b
    loss = np.sum((y_hat - y_data) ** 2) / len(x_data)
    grad = cal_gradient(x_data, y_data)
    w = w - e * grad[0]
    b = b - e * grad[1]
    print('Current Epoch: ', epoch, ' Loss: ', loss, ' w:', w, ' b:', b)
    loss_list.append(loss)
plt.plot(range(1, 51, 1), loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
