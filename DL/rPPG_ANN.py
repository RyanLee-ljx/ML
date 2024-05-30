import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class MultiData(Dataset):  # 创建数据集类，定义对数据集的一些操作，从而实例化类，即可创建数据集对象，调用数据集的属性和方法
    def __init__(self, path):
        data = np.loadtxt(path, delimiter=',', dtype=np.float32)
        self.x_data = torch.from_numpy(data[:, :4])   # 前四列
        self.y_data = torch.from_numpy(data[:, 4:])   # 后三列
        mean_x = torch.mean(self.x_data, dim=0)
        std_x = torch.std(self.x_data, dim=0)
        self.x_data = (self.x_data - mean_x)/std_x   # 标准化
        self.len = data.shape[0]  # shape返回矩阵行列，行索引为0，列为1，  本段是为了得到有多少个样本

    def __getitem__(self, item):
        return self.x_data[item, :], self.y_data[item, :]

    def __len__(self):
        return self.len


# 固定种子
seed = 404
torch.manual_seed(seed)
random.seed(seed)
# 数据建立
path = r''
batch_size = 64  # 批次大小
Data = MultiData(path)
train_data, val_data, test_data = torch.utils.data.random_split(Data, [4/5.5, 1/5.5, 0.5/5.5])
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=4)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=4)


# design model
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 因为神经网络输入要为n行一列
        self.linear1 = torch.nn.Linear(4, 12)
        self.linear2 = torch.nn.Linear(12, 36)
        self.linear3 = torch.nn.Linear(36, 108)
        self.linear4 = torch.nn.Linear(108, 324)
        self.linear5 = torch.nn.Linear(324, 24)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        y = self.activate(self.linear1(x))
        y = self.activate(self.linear2(y))
        y = self.activate(self.linear3(y))
        y = self.activate(self.linear4(y))
        y = self.linear5(y)
        return y


model = Model()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

x = torch.rand(4)
x = x.to(device)
with torch.no_grad():
    torch.onnx.export(
        model,  # 要转换的模型
        x,  # 模型的输入
        "X.onnx",  # 导出的.onnx 文件名（注意文件扩展名为.onnx）
        opset_version=11,  # ONNX 算子集版本
        input_names=["input"],  # 输入张量的名字
        output_names=["output"],  # 输出张量的名字
    )


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
            x, label = x.to(device), label.to(device)
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


def val():
    type_num = 3
    accuracy = np.zeros(type_num)
    total = 0
    str = ['Valence', 'Arouse', 'Fatigue']
    with torch.no_grad():
        for Data in val_loader:
            x_test, ground = Data
            x_test, ground = x_test.to(device), ground.to(device)
            y_pre = model(x_test)
            y_pre_percente = []
            _, valence_pre = torch.max(y_pre[:10], dim=1)  # dim=1 每行最大值  （第一个维度是行）每行是一个样本
            y_pre_percente.append(valence_pre)
            _, arouse_pre = torch.max(y_pre[10:21], dim=1)
            y_pre_percente.append(arouse_pre)
            _, fatigue_pre = torch.max(y_pre[21:-1], dim=1)
            y_pre_percente.append(fatigue_pre)
            total = total + len(ground)
            for i in range(type_num):
                accuracy[i] = accuracy[i] + (y_pre_percente[i] == ground[i]).sum().item()

        accuracy = accuracy / total  # 平均准确预测
        for i in range(type_num):
            print(str[i], f'Average accuracy in val: {accuracy[i] * 100:.2f}%')  # 将精度乘以100（*100） 并保留两位小数（:.2f）  最后加%


def test():
    type_num = 3
    accuracy = np.zeros(type_num)
    total = 0
    str = ['Valence', 'Arouse', 'Fatigue']
    with torch.no_grad():
        for Data in test_loader:
            x_test, ground = Data
            x_test, ground = x_test.to(device), ground.to(device)
            y_pre = model(x_test)
            y_pre_percente = []
            _, valence_pre = torch.max(y_pre[:10], dim=1)  # dim=1 每行最大值  （第一个维度是行）每行是一个样本
            y_pre_percente.append(valence_pre)
            _, arouse_pre = torch.max(y_pre[10:21], dim=1)
            y_pre_percente.append(arouse_pre)
            _, fatigue_pre = torch.max(y_pre[21:-1], dim=1)
            y_pre_percente.append(fatigue_pre)
            total = total + len(ground)
            for i in range(type_num):
                accuracy[i] = accuracy[i] + (y_pre_percente[i] == ground[i]).sum().item()

        accuracy = accuracy / total  # 平均准确预测
        for i in range(type_num):
            print(str[i], f'Average accuracy in test: {accuracy[i] * 100:.2f}%')  # 将精度乘以100（*100） 并保留两位小数（:.2f）  最后加%


if __name__ == '__main__':
    train(200)
    val()
    # test()
