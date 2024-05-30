import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

# prepare dataset
# 转变为张量transforms.Normalize((0.1307, ), (0.3081, ))   # 标准化])
transform = transforms.Compose((transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))))
batch_size = 64  # 批次大小
# 读取数据对象，并完成相应转化（张量化，标准化）
train_data = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
# 读取数据
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=4)


# design model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # 因为神经网络输入要为n行一列
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 64)
        self.linear5 = torch.nn.Linear(64, 10)  # 一共有10类，故最后输出其为各个概率的分布
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)        # view中一个参数定为-1，代表自动调整这个维度上的元素个数，以保证元素的总数不变。即保证每列为784，行数为N
        y = self.activate(self.linear1(x))
        y = self.activate(self.linear2(y))
        y = self.activate(self.linear3(y))
        y = self.activate(self.linear4(y))
        y = self.linear5(y)
        return y


model = Model()

# construct loss and optimizer
loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# train cycle
def train(max_epoch):
    loss_epoch = []
    for epoch in range(1, max_epoch + 1, 1):
        loss_iteration = 0
        for i, data in enumerate(train_loader, start=0):
            x, label = data
            y_hat = model(x)
            loss = loss_func(y_hat, label)
            loss_av = loss.item() / len(data)
            loss_iteration = loss_iteration + loss_av

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch.append(loss_iteration)  # 记录每个epoch loss
        if epoch % 20 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_iteration}')
    plt.plot(np.arange(1, max_epoch + 1, 1), loss_epoch)
    plt.title('Train Process')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def test():
    accuracy = 0
    total = 0
    with torch.no_grad():
        for Data in test_loader:
            image, ground = Data
            y_pre = model(image)
            print(y_pre)
            # torch.max(outputs.data, dim=1) 返回一个命名元组 (values, indices)，其中 values 是每行中的最大值，而 indices 是每个最大值的索引位置
            _, predicted = torch.max(y_pre, dim=1)  # dim=1 每行最大值  （第一个维度是行）
            # 因为输出是各类概率的分布，所以取最大的就是其属于哪一类
            total = total + len(ground)
            accuracy = accuracy + (predicted == ground).sum().item()

        accuracy = accuracy / total  # 平均准确预测
        print(f'Average accuracy: {accuracy * 100:.2f}%')   # 将精度乘以100（*100） 并保留两位小数（:.2f）  最后加%


if __name__ == '__main__':
    train(10)
    test()
