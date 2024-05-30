# 组合 Residual net & Inception
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from matplotlib import pyplot as plt

# prepare data
transform = transforms.Compose(
    (transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))))
batch_size = 32
train_data = datasets.CIFAR10(root='../dataset/CIFAR10/', train=True, download=False, transform=transform)
test_data = datasets.CIFAR10(root='../dataset/CIFAR10/', train=False, download=False, transform=transform)
# load data
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=4)


class Inception(torch.nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=24, kernel_size=1)
        self.conv1_2 = torch.nn.Conv2d(in_channels=in_channel, out_channels=16, kernel_size=1)
        self.conv5 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5, padding=2)  # 要保持原图像大小不变 故补两个0
        self.con3_1 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)  # 保持原图像大小不变 补1个
        self.con3_2 = torch.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)
        self.in_channel = in_channel

    def forward(self, x):
        branch_1 = torch.nn.functional.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_1 = self.conv1_1(branch_1)

        branch_2 = self.conv1_2(x)

        branch_3 = self.conv5(self.conv1_2(x))

        branch_4 = self.con3_2(self.con3_1(self.conv1_2(x)))

        output = [branch_1, branch_2, branch_3, branch_4]
        return torch.cat(output, dim=1)  # 维度是4维 B C W H  ，要在C上进行连接


class Residual(torch.nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.conv1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=3, padding=1)

    def forward(self, x):
        y = torch.nn.functional.relu(self.conv1(x))
        y = self.conv2(y)

        return y + x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.inception1 = Inception(32)
        self.inception2 = Inception(88)
        self.conv2 = torch.nn.Conv2d(88, 88, kernel_size=5, padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.residual1 = Residual(88)
        self.conv3 = torch.nn.Conv2d(88, 64, kernel_size=5, padding=2)
        self.residual2 = Residual(64)
        self.linear1 = torch.nn.Linear(1024, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        in_size = x.size(0)  # 获取样本数
        y = self.conv1(x)
        y = torch.nn.functional.relu(y)
        y = self.inception1(y)
        y = self.inception2(y)
        y = self.pool(y)
        y = self.conv2(y)
        y = torch.nn.functional.relu(y)
        y = self.pool(y)
        y = self.residual1(y)
        y = self.conv3(y)
        y = torch.nn.functional.relu(y)
        y = self.pool(y)
        y = self.residual2(y)
        y = y.view(in_size, -1)
        y = self.linear1(y)
        y = self.linear2(y)
        y = self.linear3(y)
        return y


model = Net()
# use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

x = torch.rand(1, 3, 32, 32)
x = x.to(device)
with torch.no_grad():
    torch.onnx.export(
        model,  # 要转换的模型
        x,  # 模型的输入
        "X-net_1.0.onnx",  # 导出的.onnx 文件名（注意文件扩展名为.onnx）
        opset_version=11,  # ONNX 算子集版本
        input_names=["input"],  # 输入张量的名字
        output_names=["output"],  # 输出张量的名字
    )

# construct loss and optimizer
loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# train cycle
def train(max_epoch):
    loss_epoch = []
    for epoch in range(1, max_epoch + 1, 1):
        loss_iteration = 0
        for i, data in enumerate(train_loader, start=0):
            x, label = data
            x, label = x.to(device), label.to(device)  # Send the inputs and targets at every step to the GPU
            y_hat = model(x)
            loss = loss_func(y_hat, label)
            loss_av = loss.item() / len(data)
            loss_iteration = loss_iteration + loss_av

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch.append(loss_iteration)  # 记录每个epoch loss
        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_iteration}')
            test()
    plt.figure(1)
    plt.plot(np.arange(1, max_epoch + 1, 1), loss_epoch)
    plt.title('Train Process')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.figure(2)
    plt.plot(np.arange(1, max_epoch/10 + 1, 1), loss_epoch)
    plt.title('Test Error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()


def test():
    accuracy = 0
    total = 0
    loss_epoch = []
    with torch.no_grad():
        for Data in test_loader:
            image, ground = Data
            image, ground = image.to(device), ground.to(device)
            y_pre = model(image)
            # torch.max(outputs.data, dim=1) 返回一个命名元组 (values, indices)，其中 values 是每行中的最大值，而 indices 是每个最大值的索引位置
            _, predicted = torch.max(y_pre, dim=1)  # dim=1 每行最大值  （第一个维度是行）
            # 因为输出是各类概率的分布，所以取最大的就是其属于哪一类
            total = total + len(ground)
            accuracy = accuracy + (predicted == ground).sum().item()

        accuracy = accuracy / total  # 平均准确预测
        print(f'Average accuracy on test data: {accuracy * 100:.2f}%')  # 将精度乘以100（*100） 并保留两位小数（:.2f）  最后加%
        loss_epoch.append(accuracy)


if __name__ == '__main__':
    train(100)
