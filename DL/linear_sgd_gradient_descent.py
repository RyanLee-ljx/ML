import numpy as np
import matplotlib.pyplot as plt

x_data = np.array([1,2,3,4])
y_data = np.array([4,7,10,13])
w = 0.01   # 初始化参数
b = 0.01   # 初始化参数
e = 0.01 # 学习率
loss_list = []
epoch = 1

def std_gradient(x,y):
    grad = []
    grad.append(2 * x * (x * w + b - y))
    grad.append(2 * (x * w + b - y))
    return grad

for x,y in zip(x_data,y_data):    # zip使得x,y成一组对
    # print("x:",x,"y:",y)
    y_hat = x * w + b
    loss = (y_hat - y) ** 2
    grad = std_gradient(x,y)
    print(grad)
    w = w - e*grad[0]
    b = b - e*grad[1]
    print('Current Epoch: ',epoch, ' Loss: ', loss, ' w:', w, ' b:', b)
    loss_list.append(loss)
    epoch = epoch +1
plt.plot(range(1,len(x_data)+1,1),loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
