# 李嘉贤 交运2104 学号：8212211218
# 代码参考：https://blog.csdn.net/qq_44853197/article/details/118275689?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-118275689-blog-113353385.235%5Ev40%5Epc_relevant_anti_t3_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-3-118275689-blog-113353385.235%5Ev40%5Epc_relevant_anti_t3_base&utm_relevant_index=4
import math
import numpy as np
import matplotlib.pyplot as plt


def initial_bp_neural_network(n_inputs, n_hidden_layers, n_outputs):
    # hidden_layers 第n层隐含层有几个神经元
    global hidden_layers_weights
    global hidden_layers_bias
    hidden_layers_bias = [np.array([-0.4, -0.2]), np.array([0.1, 0.4])]
    hidden_layers_weights=[np.array([[0.2 , 0.1], [0.4, -0.1]]), np.array([[-0.2, 0.1], [-0.1, 0.1]])]
    return hidden_layers_weights


def forward_propagate(inputs):
    global train_outputs
    global train_inputs
    train_inputs = []
    train_outputs = []
    function_vector = np.vectorize(logistic)
    for i in range(len(hidden_layers_weights)):
        if i == 0:
            outputs = np.array(hidden_layers_weights[i]).dot(test_inputs)
        else:
            outputs = np.array(hidden_layers_weights[i]).dot(inputs)
        outputs = np.array(outputs) + np.array(hidden_layers_bias[i])
        train_inputs.append(outputs)
        outputs = function_vector(outputs)
        # 记录输出
        train_outputs.append(outputs)
        inputs = outputs.copy()
        # print(inputs)
    # print(train_outputs)
    return train_outputs


def backward_error_propagate():
    global diff
    diff = []
    error_sum = RMSE_function(test_outputs, train_outputs)   # 记录损失函数
    # print(error_sum)
    function_vector = np.vectorize(logistic_diff)
    # 分别计算均方误差的导数和激活函数的导数
    for i in range(len(train_outputs)):
        if i == 0:
            diff.append(RMSE_diff_fuction(test_outputs, train_outputs[len(train_outputs) - 1]) *
                        function_vector(np.array(train_inputs[len(train_inputs) - 1])))
        else:
            diff.append(function_vector(np.array(train_inputs[len(train_inputs) - 1 - i])) *
                        hidden_layers_weights[len(train_outputs) - i].T.dot(diff[i - 1]))
    # print(diff)
    # print('\n')
    # print(len(diff))
    return error_sum


def update_weights_function():
    # 更新权重
    for i in range(len(hidden_layers_weights)):
        if i == 0:
            hidden_layers_weights[i] = hidden_layers_weights[i] - learning_rate * (
                    diff[len(diff) - i - 1].reshape(len(diff[len(diff) - i - 1]), 1) *
                    np.array(test_inputs).reshape(1, len(test_inputs)) )
        else:
            hidden_layers_weights[i] = hidden_layers_weights[i] - learning_rate * (
                    diff[len(diff) - i - 1].reshape(len(diff[len(diff) - i - 1]), 1) *
                    np.array(train_inputs[i - 1]).reshape(1, len(train_inputs[i - 1])) )
    for i in range(len(hidden_layers_bias)):
        hidden_layers_bias[i] = hidden_layers_bias[i] - learning_rate * (diff[len(diff) - i - 1])
    # print(hidden_layers_weights)
    # print(hidden_layers_bias)
    # print('\n')


# 均方误差函数
def RMSE_function(actual_outputs, predict_outputs):
    function_vector = np.vectorize(pow)
    # print(np.array(actual_outputs))
    # print(np.array(predict_outputs)[1])
    m = 1 / 2 * (sum(function_vector(np.array(actual_outputs) - np.array(predict_outputs)[1], 2)))
    # print(m)
    # print('\n')
    return m


# 均方误差函数的导数
def RMSE_diff_fuction(actual_outputs, predict_outputs):
    return np.array(predict_outputs)-np.array(actual_outputs)


# 激活函数
def logistic(x):
    return 1 / (1 + math.exp(-x))


# 激活函数logistic sigmoid函数的导数
def logistic_diff(x):
    return logistic(x) * (1 - logistic(x))


def finish():
    for i in range(len(train_outputs[len(train_outputs) - 1])):
        if abs(train_outputs[len(train_outputs) - 1][i] - test_outputs[i]) > 0.01:         #收敛精度定为1%
            return 0
    return 1


def train_network(iteration):
    results = []
    for i in range(iteration):
        forward_propagate(test_inputs)
        result = backward_error_propagate()
        # print(result)
        results.append(result)
        # print(results)
        update_weights_function()
        if finish() == 1:
            return results


if __name__ == '__main__':
    learning_rate = 0.3   # 学习率
    test_inputs = [1, 0]   # 输入
    test_outputs = [0.9, 0.1]   # tag
    initial_bp_neural_network(2, [2], 2)  # 初始化BP神经网络
    error_data = train_network(4000)   # 训练
    # print(error_data)
    # 绘图
    plt.plot(range (1, len(error_data)+1), error_data)
    plt.xlabel('number of Iteration')
    plt.ylabel('Loss Function')
    plt.title('Iteration')
    plt.show()
