import torch

x_data = [1.0, 2.0, 3.0, 4.0]
y_data = [3.0, 7.0, 13.0, 21.0]
w1 = torch.tensor([1.0])
w2 = torch.tensor([1.0])
b = torch.tensor([1.0])
w1.requires_grad = True
b.requires_grad = True
w2.requires_grad = True
max_epoch = 30
e = 0.01


def forward(x):
    return w1 * x ** 2 + w2 * x + b


def loss(x, y):
    y_hat = forward(x)
    loss_val = (y - y_hat) ** 2
    return loss_val


for epoch in range(1, max_epoch + 1, 1):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        # print(l)
        l.backward()
        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item())
        w1.data = w1.data - e * w1.grad.data
        w2.data = w2.data - e * w2.grad.data
        b.data = b.data - e * b.grad.data

        w1.grad.data.zero_()  # 更新完一次后梯度清零
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print('\tEpoch: ', epoch, w1.item(), w2.item(), b.item())
