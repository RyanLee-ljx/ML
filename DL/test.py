# import numpy as np
# a = [2, 3, 6, 8, 9, 0, 2]
# print(a[:4])
# print(a[4:6])
# type_num = 3
# accuracy = np.zeros(type_num)
# total = 0
# for i in range(type_num):
#     print(i)
#     accuracy[i] = accuracy[i] + (a[i] == 1)
# accuracy = accuracy/3
# str = ['Valence', 'Arouse', 'Fatigue']
# for i in range(type_num):
#     print(str[i], f'Average accuracy: {accuracy[i] * 100:.2f}%')  # 将精度乘以100（*100） 并保留两位小数（:.2f）  最后加%


import torch

# Set print options
torch.set_printoptions(linewidth=2048, precision=6, sci_mode=False)

# Define a tensor
a = torch.tensor([[1.00, 10.05, 1.21, 1.24, 1.38, 1.39, 1.78, 1.81, 1.99],
                  [2.00, 23.04, 2.21, 2.22, 2.36, 2.37, 2.77, 2.83, 2.94],
                  [3.00, 30.06, 3.20, 3.26, 3.32, 3.33, 3.76, 3.84, 3.98],
                  [4.00, 42.08, 4.21, 4.28, 4.37, 4.38, 4.79, 4.88, 4.98],
                  [5.00, 52.01, 5.22, 5.22, 5.35, 5.35, 5.72, 5.84, 5.96],
                  [6.00, 62.02, 6.24, 6.25, 6.33, 6.33, 6.71, 6.86, 6.99],
                  [7.00, 72.01, 7.26, 7.24, 7.32, 7.33, 7.71, 7.89, 7.95],
                  [8.00, 83.01, 8.25, 8.26, 8.31, 8.34, 8.70, 8.89, 8.96]])
# Define a contrastive tensor
b = torch.tensor([1.00, 1.05, 1.21, 1.24, 1.38, 1.39, 1.78, 1.81, 1.99])

mean_a = torch.mean(a, dim=0)
std_a = torch.std(a, dim=0)
n_a = (a - mean_a)/std_a
print(n_a)




# # Calculate mean and standard variance
# mean_a = torch.mean(a, dim=1)
# mean_b = torch.mean(b, dim=0)
# print(mean_a)
# print(mean_b)
# std_a = torch.std(a, dim=1)
# std_b = torch.std(b, dim=0)
# print(std_a)
# print(std_b)
#
# # Do Z-score standardization on 2D tensor
# n_a = a.sub_(mean_a[:, None]).div_(std_a[:, None])
# n_b = (b - mean_b) / std_b
# print(n_a)
# print(n_b)