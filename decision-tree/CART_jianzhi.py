# 李嘉贤 交运2104 学号：8212211218
# 代码参考：https://blog.csdn.net/m0_46345193/article/details/125310738?spm=1001.2014.3001.5502
import copy
from collections import Counter

import numpy as np
import pandas as pd

import plotTree as pT


# 计算数据集的信息熵(Information Gain)
def calc_InfoEnt(dataset):  # dataset每一列是一个属性(列末是label)
    num_entries = len(dataset)  # dataset每一行是一个样本
    label_counts = {}  # 给所有可能的分类创建字典label_counts
    for featVec in dataset:  # 按行循环
        current_label = featVec[-1]  # featVec的最后一个值为label
        if current_label not in label_counts.keys():  # 如果当前label还未在字典中出现
            label_counts[current_label] = 0  # 创建该label的key
        label_counts[current_label] += 1  # 统计每一类label的数量
    InfoEnt = 0.0  # 初始化InfoEnt信息熵的值
    for key in label_counts:
        p = float(label_counts[key]) / num_entries  # 求出每一类label的概率
        InfoEnt -= p * np.math.log(p, 2)  # 信息熵计算公式
    return InfoEnt


def split_discrete_dataset(dataset, feature_index, value):
    # dataset:当前结点(待划分)集合, axis:指示划分所依据的属性, value:该属性用于划分的取值
    dataset_out = []  # 为return dataset 返回一个列表
    for featVec in dataset:  # 抽取符合条件的特征值
        if featVec[feature_index] == value:
            reduced_feat = featVec[:feature_index]  # 该特征之前的特征仍然保留在dataset中
            reduced_feat.extend(featVec[feature_index + 1:])  # 该特征之后的特征仍然保留在样本中
            dataset_out.append(reduced_feat)  # 把去除掉axis特征的样本加入到list
    return dataset_out


def split_continuous_dataset(dataset, feature_index, value):
    dataset_out_0 = []
    dataset_out_1 = []
    for featVec in dataset:
        if featVec[feature_index] > value:
            reduced_feat_1 = featVec[:feature_index]  # 该特征之前的特征仍然保留在dataset中
            reduced_feat_1.extend(featVec[feature_index + 1:])  # 该特征之后的特征仍然保留在样本中
            dataset_out_1.append(reduced_feat_1)
        else:
            reduced_feat_0 = featVec[:feature_index]  # 该特征之前的特征仍然保留在dataset中
            reduced_feat_0.extend(featVec[feature_index + 1:])  # 该特征之后的特征仍然保留在样本中
            dataset_out_0.append(reduced_feat_0)
    return dataset_out_0, dataset_out_1  # 返回两个集合，分别为大于和小于该value


# 根据InfoGain选择当前最好的划分特征(以及对于连续变量还要选择以什么值划分)
def ID3_best_split(dataset, label):
    feat_num = len(dataset[0]) - 1  # 根据dataset判断要划分的特征的数量
    base_Ent = calc_InfoEnt(dataset)  # 计算初始Ent
    best_infoGain = 0.0  # 初始化信息增益率
    best_feature = -1
    best_split = -1
    best_split_dict = {}
    for i in range(feat_num):
        # 遍历所有特征：取每一行的第i个，即得当前集合所有样本第i个feature的值
        feat_list = [example[i] for example in dataset]
        # 判断是否为离散特征
        if not (type(feat_list[0]).__name__ == 'float' or type(feat_list[0]).__name__ == 'int'):
            # 对于离散特征：求若以该特征划分的增熵
            unique_vals = set(feat_list)  # 从列表中创建集合set(获得得列表唯一元素值)
            new_Ent = 0.0
            for value in unique_vals:  # 遍历该离散特征每个取值
                sub_dataset = split_discrete_dataset(dataset, i, value)  # 计算每个取值的熵
                p = len(sub_dataset) / float(len(dataset))
                new_Ent += p * calc_InfoEnt(sub_dataset)  # 各取值的熵累加
            infoGain = base_Ent - new_Ent  # 得到以该特征划分的熵增
            # 对于连续特征：求若以该特征划分的增熵(n个数据需要添加n-1个候选划分点，并选择最佳划分点)
        else:
            # 产生n-1个候选划分点
            sort_feat_list = sorted(feat_list)
            split_list = []
            for j in range(len(sort_feat_list) - 1):  # 产生n-1个候选划分点
                split_list.append(round(((sort_feat_list[j] + sort_feat_list[j + 1]) / 2.0), 3))
            best_split_Ent = 10000
            # 遍历n-1个候选划分点：求第j个候选划分点划分时的增熵，并选择最佳划分点
            for j in range(len(split_list)):
                value = split_list[j]
                new_Ent = 0.0
                new_dataset = split_continuous_dataset(dataset, i, value)
                sub_dataset_0 = new_dataset[0]
                sub_dataset_1 = new_dataset[1]
                p0 = len(sub_dataset_0) / float(len(dataset))
                new_Ent += p0 * calc_InfoEnt(sub_dataset_0)
                p1 = len(sub_dataset_1) / float(len(dataset))
                new_Ent += p1 * calc_InfoEnt(sub_dataset_1)
                if new_Ent < best_split_Ent:
                    best_split_Ent = new_Ent
                    best_split = j
            best_split_dict[label[i]] = split_list[best_split]  # 字典记录当前连续属性的最佳划分点
            infoGain = base_Ent - best_split_Ent  # 计算以该节点划分的熵增
        # 在所有属性(包括连续和离散)中选择可以获得最大熵增的属性
        if infoGain >= best_infoGain:
            best_infoGain = infoGain
            best_feature = i
    # 若当前节点的最佳划分特征为连续特征，则需根据“是否小于等于其最佳划分点”进行二值化处理
    if type(dataset[0][best_feature]).__name__ == 'float' or \
            type(dataset[0][best_feature]).__name__ == 'int':
        best_split_value = best_split_dict[label[best_feature]]
        label[best_feature] = label[best_feature] + '<=' + str(best_split_value)
        for i in range(np.shape(dataset)[0]):
            if dataset[i][best_feature] <= best_split_value:
                dataset[i][best_feature] = 1
            else:
                dataset[i][best_feature] = 0
    return best_feature


# 若特征已经划分完，节点下的样本还没有统一取值，则需要进行投票：计算每类Label个数, 取max者
def majorityCnt(classList):
    class_count = {}  # 将创建键值为Label类型的字典
    for vote in classList:
        if vote not in class_count.keys():
            class_count[vote] = 0  # 第一次出现的Label加入字典
        class_count[vote] += 1  # 计数
    return max(class_count)


def CART_best_split(dataset):
    feat_num = len(dataset[0]) - 1  # 根据dataset判断要划分的特征的数量
    best_Gini = 99999.0  # 初始化Gini指数
    best_feature = -1
    for i in range(feat_num):
        # 遍历所有特征：取每一行的第i个，即得当前集合所有样本第i个feature的值
        feat_list = [example[i] for example in dataset]
        unique_vals = set(feat_list)  # 从列表中创建集合set(获得得列表唯一元素值)
        gini = 0.0
        for value in unique_vals:
            sub_dataset = split_discrete_dataset(dataset, i, value)  # 计算每个取值的熵
            p = len(sub_dataset) / float(len(dataset))
            sub_p = len(split_discrete_dataset(sub_dataset, -1, '否')) / float(len(sub_dataset))
            # gini += p * (1.0 - pow(sub_p, 2) - pow(1 - sub_p, 2))
            gini += 2 * p * sub_p * (1 - sub_p)
        print(u"CART中第%d个特征的基尼值为：%.3f" % (i, gini))
        if gini < best_Gini:
            best_Gini = gini
            best_feature = i
    return best_feature


# 递归产生决策树
def ID3_createTree(dataset, labels, data_full, labels_full, test_dataset):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        # 类别完全相同，停止划分
        return class_list[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(class_list)
    best_feat = ID3_best_split(dataset, labels)
    best_featLabel = labels[best_feat]
    print(u"此时最优索引为：" + best_featLabel)
    ID3_Tree = {best_featLabel: {}}
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)
    unique_vals_full = set()
    if type(dataset[0][best_feat]).__name__ == 'str':
        current_label = labels_full.index(labels[best_feat])
        feat_values_full = [example[current_label] for example in data_full]
        unique_vals_full = set(feat_values_full)
    del (labels[best_feat])  # 划分完后, 即当前特征已经使用过了, 故将其从“待划分特征集”中删去
    # 【递归调用】针对当前用于划分的特征(beat_Feat)的每个取值，划分出一个子树
    for value in unique_vals:  # 遍历该特征【现存的】取值
        sub_labels = labels[:]
        if type(dataset[0][best_feat]).__name__ == 'str':
            unique_vals_full.remove(value)  # 划分后删去(从unique_Vals_Full中删!)
        ID3_Tree[best_featLabel][value] = ID3_createTree(split_discrete_dataset(dataset, best_feat, value), sub_labels,
                                                         data_full, labels_full, test_dataset)
    if type(dataset[0][best_feat]).__name__ == 'str':
        for value in unique_vals_full:
            ID3_Tree[best_featLabel][value] = majorityCnt(class_list)

    return ID3_Tree


def CART_creatTree(dataset, labels, test_dataset):
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        # 类别完全相同，停止划分
        return class_list[0]
    if len(dataset[0]) == 1:
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(class_list)
    best_feat = CART_best_split(dataset)
    best_featLabel = labels[best_feat]
    print(u"此时最优索引为：" + best_featLabel)
    CART_Tree = {best_featLabel: {}}
    del (labels[best_feat])
    # 得到列表包括节点所有的属性值
    feat_values = [example[best_feat] for example in dataset]
    unique_vals = set(feat_values)

    if pre_pruning is True:
        ans = []
        for index in range(len(test_dataset)):
            ans.append(test_dataset[index][-1])
        result_counter = Counter()
        for vec in dataset:
            result_counter[vec[-1]] += 1
        leaf_output = result_counter.most_common(1)[0][0]
        root_acc = cal_acc(test_output=[leaf_output] * len(test_dataset), label=ans)
        outputs = []
        ans = []
        for value in unique_vals:
            cut_testSet = split_discrete_dataset(test_dataset, best_feat, value)
            cut_dataSet = split_discrete_dataset(dataset, best_feat, value)
            for vec in cut_testSet:
                ans.append(vec[-1])
            result_counter = Counter()
            for vec in cut_dataSet:
                result_counter[vec[-1]] += 1
            leaf_output = result_counter.most_common(1)[0][0]
            outputs += [leaf_output] * len(cut_testSet)
        cut_acc = cal_acc(test_output=outputs, label=ans)

        if cut_acc <= root_acc:
            return leaf_output

    # 针对当前用于划分的特征(beat_Feat)的每个取值，划分出一个子树
    for value in unique_vals:  # 遍历该特征【现存的】取值
        sub_labels = labels[:]
        CART_Tree[best_featLabel][value] = CART_creatTree(split_discrete_dataset(dataset, best_feat, value),
                                                          sub_labels,
                                                          split_discrete_dataset(test_dataset, best_feat, value))

    return CART_Tree


def classify(input_tree, feat_labels, test_Vec):
    class_label = ''
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)
    for key in second_dict.keys():
        if test_Vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_Vec)
            else:
                class_label = second_dict[key]

    return class_label


# 测试决策树正确率
def tree_acc(input_tree, data_test, labels):
    error = 0.0
    for i in range(len(data_test)):
        if classify(input_tree, labels, data_test[i]) != data_test[i][-1]:
            error += 1

    return float(error)


# 测试投票节点正确率
def major_acc(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1

    return float(error)


def post_pruning(input_tree, dataset, data_test, labels):
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    class_list = [example[-1] for example in dataset]
    feat_key = copy.deepcopy(first_str)
    label_index = labels.index(feat_key)
    temp_labels = copy.deepcopy(labels)
    del(labels[label_index])
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            if type(dataset[0][label_index]).__name__ == 'str':
                input_tree[first_str][key] = post_pruning(second_dict[key],
                                                          split_discrete_dataset(dataset, label_index, key),
                                                          split_discrete_dataset(data_test, label_index, key),
                                                          copy.deepcopy(labels))

    if tree_acc(input_tree, data_test, temp_labels) <= major_acc(majorityCnt(class_list), data_test):
        return input_tree
    return majorityCnt(class_list)


def cal_acc(test_output, label):
    assert len(test_output) == len(label)
    count = 0
    for index in range(len(test_output)):
        if test_output[index] == label[index]:
            count += 1

    return float(count / (len(test_output)))


def main():
    # train_df = pd.read_csv('watermelon.txt', encoding='gbk', sep=',')
    train_df = pd.read_csv('Train_set.txt',encoding='utf-8',sep=',')     # 读取训练集
    data = train_df.values[0:, 1:].tolist()   # 训练集
    test_df = pd.read_csv('Validation_set.txt', encoding='utf-8',sep=',')   # 读取测试集
    test = test_df.values[0:, 1:].tolist()    # 测试集
    labels = train_df.columns.values[1:-1].tolist()    # 标签
    labels_full = labels[:]
    # print(data)
    # print('\n')
    # print(len(data))
    # print(test)
    # print('\n')
    # print(len(test))
    myTree = CART_creatTree(data, labels, test)
    # print(myTree)
    pT.createPlot(myTree)
    myTree = post_pruning(myTree, data, test, labels_full)
    print('myTree', myTree)
    pT.createPlot(myTree)


if __name__ == '__main__':
    pre_pruning = False     # 是否进行预剪枝
    main()
