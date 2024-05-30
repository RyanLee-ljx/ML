# 本代码实现糖尿病多特征二分类  第6-8节网络模型

"""
分类问题本身是维度的降低，中间神经元的作用就是逐步降低维度
多分类问题采取sigmoid外的激活函数时，最后一层最好用sigmoid输出，避免问题
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from matplotlib import pyplot as plt


# prepare data
class DiabetesDataset(Dataset):  # 创建数据集类，定义对数据集的一些操作，从而实例化类，即可创建数据集对象，调用数据集的属性和方法
    def __init__(self, path):
        xy = np.loadtxt(path, delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

        self.len = xy.shape[0]  # shape返回矩阵行列，行索引为0，列为1，  本段是为了得到有多少个样本

    def __getitem__(self, item):
        return self.x_data[item, :], self.y_data[item, :]

    def __len__(self):
        return self.len


# 实例化数据集对象
Data = DiabetesDataset('diabetes.csv.gz')
train_data = DataLoader(dataset=Data, batch_size=32, shuffle=True, num_workers=4)


# design model


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 2)
        self.linear4 = torch.nn.Linear(2, 1)
        self.activate1 = torch.nn.ReLU()
        self.activate2 = torch.nn.Sigmoid()

    def forward(self, x):
        y = self.linear1(x)
        y = self.activate1(y)
        y = self.linear2(y)
        y = self.activate1(y)
        y = self.linear3(y)
        y = self.activate1(y)
        y = self.linear4(y)
        y = self.activate2(y)
        return y


model = Model()  # 实例化模型

# construct loss and optimizer
'''
在pytorch中nn.CrossEntropyLoss()为交叉熵损失函数，用于解决多分类问题，也可用于解决二分类问题。
BCELoss是Binary CrossEntropyLoss的缩写，nn.BCELoss()为二元交叉熵损失函数，只能解决二分类问题。
在使用nn.BCELoss()作为损失函数时，需要在该层前面加上Sigmoid函数，一般使用nn.Sigmoid()即可，
'''
Loss_func = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# train cycle

if __name__ == '__main__':
    loss_epoch = []  # every epoch loss
    max_epoch = 50
    for epoch in range(1, max_epoch + 1, 1):
        loss_iteration = 0  # every iteration loss
        for i, data in enumerate(train_data, 0):  # 按开始索引遍历索引数据集 enumerate(data, start_num)
            x, label = data
            y_hat = model(x)  # forward
            loss = Loss_func(y_hat, label)  # loss
            loss_iteration = loss_iteration + loss.item()

            optimizer.zero_grad()
            loss.backward()  # backward
            optimizer.step()  # update
        loss_iteration = loss_iteration / Data.len
        print(f'Current epoch: {epoch}, Loss: {loss_iteration}')
        loss_epoch.append(loss_iteration)
    plt.plot(np.arange(1, max_epoch + 1, 1), loss_epoch)
    plt.title('Epoch Process')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
