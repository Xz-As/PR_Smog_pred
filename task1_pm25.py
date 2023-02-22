import pandas as pd
import numpy as np
import math

# 数据预处理
def dataProcess(df, typ):
    x_list, y_list = [], []
    # df替换指定元素，将空数据填充为0
    df = df.replace(['NR'], [0.0])
    # astype() 转换array中元素数据类型
    array = np.array(df).astype(float)
    # 将数据集拆分为多帧
    for i in range(0, len(array), 18):#共3240行，每18行为一天的数据
        for j in range(typ - 9):#每天共typ个小时
            mat = array[(i+8, i+9, i+12), j:j + 9]
            x_list.append(mat)
            if typ == 24:
                label = array[i + 9, j + 9]  # 第10行是PM2.5,第10小时是预测数据（不知道起始时间，只好每个时间段都学习）
                y_list.append(label)
    x = np.array(x_list)
    y = np.array(y_list)
    return x, y, array


# 计算绝对值
def abs_(ori:float=0):
    return (ori if ori > 0.0 else -ori)


# 更新参数，训练模型
def train(x_train, y_train, epoch):
    bias = 0  # 偏置初始化
    weights = np.ones(9)  # 权重初始化
    #weights = np.random.rand(9) # 随机初始权重
    learning_rate = 9000  # 初始学习率
    reg_rate = 0.001  # 正则项系数
    bg2_sum = 0  # 偏置梯度平方和
    wg2_sum = np.zeros(9)  # 初始化权重梯度平方和
    l1s =  []
    l2s = []
    nses = []
    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(9)
        # 在所有数据上计算Loss_label的梯度
        for j in range(len(x_train)):
            b_g += (y_train[j] - weights.dot(x_train[j, 1, :]) - bias) * (-1)
            for k in range(9):
                w_g[k] += (y_train[j] - weights.dot(x_train[j, 1, :]) - bias) * (-x_train[j, 1, k])
        # 求平均    
        b_g /= len(x_train)
        w_g /= len(x_train)

        #  加上Loss_regularization在w上的梯度
        for m in range(9):
            w_g[m] += reg_rate * weights[m]

        # addgrad
        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2
        # 更新权重和偏置
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g

        # 训练集上loss(100epoch/次)
        if (i + 1) % 100 == 0:
            l1 = L1_(x_train[:, 1], y_train, weights, bias)
            l2 = L2_(x_train[:, 1], y_train, weights, bias)
            NSE = NSE_(x_train[:, 1], y_train, weights, bias)
            l1s.append(l1)
            l2s.append(l2)
            nses.append(NSE)
            print(f'epoch = {i + 1}\nl1 loss = {l1}\nl2 loss = {l2}\nNSE = {NSE}')

    return weights, bias, l1s, l2s, nses


# 计算L1
def L1_(x, y, w, b):
    l1 = 0
    for i in range(len(x)):
        l1 +=  abs_(y[i] - (w.dot(x[i]) + b))
    return l1 / len(x)


# 计算L2
def L2_(x, y, w, b):
    l2 = 0
    for i in range(len(x)):
        l2 +=  (y[i] - (w.dot(x[i]) + b)) ** 2
    return l2 / len(x)


def NSE_(x, y, w, b):
    sim = np.zeros(len(x))
    for i in range(len(x)):
        sim[i] = w.dot(x[i] + b)
    denominator = np.sum((y - np.mean(y)) ** 2)
    numerator = np.sum((sim - y) ** 2)
    nse_val = 1 - numerator / denominator
    return nse_val


# 验证模型效果
def pred(x_val, weights, bias):
    y_val = np.zeros(len(x_val))
    for j in range(len(x_val)):
        y_pred = weights.dot(x_val[j, 1, :]) + bias
        y_val[j] = y_pred
    tx = ''
    print(len(y_val))
    for i in range(len(y_val)):
        tx += str(int(y_val[i] + 0.5))+'\n'
    return y_val, tx


def main():
    # 从csv中读取有用的信息
    df = pd.read_csv('train.csv', usecols=range(3, 27), encoding = 'gbk')
    x, y, _ = dataProcess(df, 24)
    # 划分训练集与验证集
    x_train, y_train = x, y
    epoch = 500  # 训练轮数
    # 开始训练
    w, b, l1s, l2s, nses = train(x_train, y_train, epoch)
    # 测试集
    df = pd.read_csv('test1.csv', usecols=range(2, 2+9), encoding = 'gbk')
    x, y, _ = dataProcess(df, 10)
    print(len(x))
    y_t, tx = pred(x, w, b)
    for i in range(len(l1s)):
        tx += str(l1s[i]) + ',' + str(l2s[i]) + ',' + str(nses[i]) + '\n'
    with open('sub.csv', 'w') as f:
        f.write(tx)
    #print('The loss on val data is:', loss)


if __name__ == '__main__':
    main()