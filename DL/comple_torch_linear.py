import torch

# prepare dataset
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor(([[5.0], [9.0], [13.0]]))
max_epoch = 200

# design model using class
class Linearmodel(torch.nn.Module):  # 继承自  torch.nn.Module
    def __init__(self):  # 初始化
        super(Linearmodel, self).__init__()  # 继承父类所有方法
        self.linear = torch.nn.Linear(1,1)  # 创建线性模型对象,input_size和output_size都是一维

    def forward(self, x):  # 重写向前传播函数
        return self.linear(x)  # 调用对象,nn.Module内所有对象都是可call的，因此该类的对象都是可以像函数一样调用
        # 调用后，通过该类的__call__函数，调用forward()


model = Linearmodel()  # 实例化，创建一个线型模型对象


loss_fuc = torch.nn.MSELoss(size_average=False)    # design loss funciton
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)    # choose optimizing approach   model.parameters:针对所有参数


for epoch in range(1,max_epoch+1,1):
    y_hat = model(x_data)   # 向前传播，通过调用对象实现
    loss = loss_fuc(y_hat,y_data)   # 计算损失
    print('epoch: ', epoch, 'loss: ', loss.item())

    optimizer.zero_grad()   # 再每次反向传播时要将梯度更新为0，避免梯度累加
    loss.backward()   # 反向传播
    optimizer.step()  # 更新参数

    print('epoch: ',epoch, 'w: ', model.linear.weight.item(), 'b:', model.linear.bias.item())
    # 先调用对象linearmodel的对象model内的linear属性，而linear本身又是一个对象，再调用其weight,bias属性，而且二者均为tensor，调用item提取其数值


x_test = torch.tensor([4.0])
y_test = torch.tensor([17.0])
print('y_pred: ',model(x_test),'MAE: ', abs(model(x_test) - y_test))