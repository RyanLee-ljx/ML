# 李嘉贤 交运2104 学号：8212211218
# 代码参考：https://blog.csdn.net/m0_46345193/article/details/125310738?spm=1001.2014.3001.5502
import pandas as pd
import plotTree as pT # 导入plotTree.py文件
from  collections import Counter


# 对离散变量进行划分
def split_discrete_dataset(dataset, feature_index, value):
    # dataset:当前结点(待划分)集合, axis:指示划分所依据的属性, value:该属性用于划分的取值
    dataset_out = []  # 为return dataset 返回一个列表
    for featVec in dataset:  # 抽取符合条件的特征值
        if featVec[feature_index] == value:
            reduced_feat = featVec[:feature_index]  # 该特征之前的特征仍然保留在dataset中
            reduced_feat.extend(featVec[feature_index + 1:])  # 该特征之后的特征仍然保留在样本中
            dataset_out.append(reduced_feat)  # 把去除掉axis特征的样本加入到list
    return dataset_out


# 根据Gini选择当前最好的划分特征
def CART_best_split(dataset):
    feat_num = len(dataset[0]) - 1  # 根据dataset判断要划分的特征的数量
    best_Gini = 9999.0  # 初始化Gini指数
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
        # print(u"CART中第%d个特征的基尼值为：%.3f" % (i, gini))
        if gini < best_Gini:
            best_Gini = gini
            best_feature = i
    return best_feature


# 若特征已经划分完，节点下的样本还没有统一取值，则需要进行投票：计算每类Label个数, 取max者
def majorityCnt(classList):
    class_count = {}  # 将创建键值为Label类型的字典
    for vote in classList:
        if vote not in class_count.keys():
            class_count[vote] = 0  # 第一次出现的Label加入字典
        class_count[vote] += 1  # 计数
    return max(class_count)


# 递归产生决策树
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
    print(u"此时最优索引为："+str(best_featLabel))
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

    # 【递归调用】针对当前用于划分的特征(beat_Feat)的每个取值，划分出一个子树
    for value in unique_vals:  # 遍历该特征【现存的】取值
        sub_labels = labels[:]
        CART_Tree[best_featLabel][value] = CART_creatTree(split_discrete_dataset(dataset, best_feat, value),
                                                          sub_labels,
                                                          split_discrete_dataset(test_dataset, best_feat, value))

    return CART_Tree


def cal_acc(test_output, label):
    assert len(test_output) == len(label)
    count = 0
    for index in range(len(test_output)):
        if test_output[index] == label[index]:
            count += 1

    return float(count / (len(test_output)))


def classify(input_Tree, feat_labels, test_Vec):
    first_str = list(input_Tree.keys())[0]
    second_dict = input_Tree[first_str]
    feat_index = feat_labels.index(first_str)
    class_label = ''
    for key in second_dict.keys():
        if test_Vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_Vec)
            else:
                class_label = second_dict[key]
    return class_label


def classifyTest(input_Tree, feat_labels, test_dataset):
    class_label_all = []
    for test_Vec in test_dataset:
        class_label_all.append(classify(input_Tree, feat_labels, test_Vec))
    return class_label_all


def main():
    df = pd.read_csv('watermelon.txt', sep=',', encoding='gbk')  # 读取txt文件
    data = df.values[0:, 1:].tolist()
    print(data)
    print('\n')
    test = df.values[10:, 1:].tolist()
    labels = df.columns.values[1:-1].tolist()
    myTree = CART_creatTree(data, labels, test)
    print(data)
    print(len(data))
    # print('myTree', myTree)
    pT.createPlot(myTree)


if __name__ == '__main__':
    pre_pruning = False     # 不进行剪枝操作
    main()
