from pandas import read_csv as csv
import numpy as np


COLUMNS = ['AMB_TEMP','CH4','CO','NMHC','NO','NO2','NOx','O3','PM10','PM2.5','RAINFALL','RH','SO2','THC','WD_HR','WIND_DIREC','WIND_SPEED','WS_HR']


"""数据存储方式是每月：{col : [] * 18}"""
def data_preprocessing(df, df1):
    year_ = {}
    month_ = {}
    months = []
    for i in range(18):
        month_[COLUMNS[i]] = []
        year_[COLUMNS[i]] = []
    month0 = month_
    # df替换指定元素，将空数据填充为0
    df = df.replace(['NR'], [0.0])
    # astype() 转换array中元素数据类型
    array = np.array(df).astype(float)
    print(f'got {len(array)} training data')
    # 将数据集拆分为多帧
    for i in range(0, len(array), 15 * 18):#每月15天
        for j in range(0, 15 * 18, 18):#每天18行
            for k in range(18):
                for l in range(24):
                    month_[COLUMNS[k]].append(array[i + j + k, l])
                    year_[COLUMNS[k]].append(array[i + j + k, l])
        months.append(month_)
        month_ = month0

    df1 = df1.replace(['NR'], [0.0])
    array1 = np.array(df1).astype(float)
    for j in range(0, len(array1), 18):#每天18行
        for k in range(18):
            for l in range(9):
                year_[COLUMNS[k]].append(array1[j + k, l])

    tx = ','.join(COLUMNS)
    for j in range(len(year_['PM2.5'])):
        tx += '\n'
        for i in range(17):
            tx += str(year_[COLUMNS[i]][j]) + ','
        tx += str(year_[COLUMNS[17]][j])
    with open('data_pre.csv', 'w') as f:
        f.write(tx)
    return months, year_


def _guiyihua(datas:dict = {}):
    """
    这里使用的是max-min归一化，所有结果将会被映射到[0, 1]间，且pm2.5的归一化指标将会被记录
    """
    if datas == {}:#数据检查
        raise RuntimeError('Data has no columns')
    data_m = {}
    for i in datas.keys():
        l1 = datas[i]
        if type(l1) != list:
            raise RuntimeError(f'the column {str(i)} is not a list, data type error')
        mx = max(l1)
        mn = min(l1)
        mx -= mn
        if str(i) == 'PM2.5':
            pm25_mn = mn
            pm25_mx = mx
        for j in range(len(l1)):
            l1[j] -= mn
            l1[j] /= mx
        data_m[str(i)] = l1

    tx = ','.join(COLUMNS)
    for j in range(len(data_m['PM2.5'])):
        tx += '\n'
        for i in range(17):
            tx += str(data_m[COLUMNS[i]][j]) + ','
        tx += str(data_m[COLUMNS[17]][j])
    with open('data_g.csv', 'w') as f:
        f.write(tx)    
    with open('data_gscale.txt', 'w') as f:
        f.write('mx,mn\n' + str(pm25_mx) + ',' + str(pm25_mn))
    return data_m, pm25_mx, pm25_mn


def data_div(root):
    df = csv(root)
    train = {}
    test = {}
    val = {}
    for i in df.keys():
        train[i] = []
        test[i] = []
        val[i] = []
    for j in range(len(df[df.keys()[0]])):
        for i in df.keys():
            if j < 4320 / 12 * 11:
                train[i].append(df[i][j])
            elif j < 4320:
                val[i].append(df[i][j])
            else:
                test[i].append(df[i][j])
    print(len(train['a']))
    test_sam = {}
    for i in train.keys():
        if i != 'PM2.5':
            test_sam[i] = []
    for i in range(9):
        test_sam['PM2.5_' + str(i + 1)] = []
    
    train_sam = {}
    for i in train.keys():
        if i != 'PM2.5':
            train_sam[i] = []
    for i in range(10):
        train_sam['PM2.5_' + str(i + 1)] = []
    
    val_sam = {}
    for i in train.keys():
        if i != 'PM2.5':
            val_sam[i] = []
    for i in range(10):
        val_sam['PM2.5_' + str(i + 1)] = []

    print(len(test_sam['a']))
    for j in range(0, len(train['a']), 15 * 24):#每月15*24个样本点
        for k in range(9, 15 * 24):#每月前9个点不能做预测
            if j + k == 3960:
                print(j, k)
            for i in train.keys():
                if i != 'PM2.5':
                    train_sam[i].append(train[i][j + k - 1])
            for i in range(10):
                train_sam['PM2.5_' + str(i + 1)].append(train['PM2.5'][j + k - 9 + i])

    for k in range(9, 15 * 24):#每月前9个点不能做预测
        for i in val.keys():
            if i != 'PM2.5':
                val_sam[i].append(val[i][k - 1])
        for i in range(10):
            val_sam['PM2.5_' + str(i + 1)].append(val['PM2.5'][k - 9 + i])

    print(len(test_sam['a']))
    for j in range(0, len(test['a']), 9):
        for i in test.keys():
            if i != 'PM2.5':
                test_sam[i].append(test[i][j + 8])
        for i in range(9):
            test_sam['PM2.5_' + str(i + 1)].append(test['PM2.5'][j + i])
    
    print(len(test_sam['a']))
    tx_ = ','.join(test_sam.keys())
    tx_t = tx_ + ',PM2.5_10'
    tx_v = tx_t
    for j in range(len(test_sam['a'])):
        tx_ += '\n'
        for i in range(len(test_sam.keys())):
            if i == 0:
                tx_ += str(test_sam[list(test_sam.keys())[i]][j])
            else:
                tx_ += ',' + str(test_sam[list(test_sam.keys())[i]][j])
    for j in range(len(train_sam['a'])):
        tx_t += '\n'
        for i in range(len(train_sam.keys())):
            if i == 0:
                tx_t += str(train_sam[list(train_sam.keys())[i]][j])
            else:
                tx_t += ',' + str(train_sam[list(train_sam.keys())[i]][j])
    for j in range(len(val_sam['a'])):
        tx_v += '\n'
        for i in range(len(val_sam.keys())):
            if i == 0:
                tx_v += str(val_sam[list(val_sam.keys())[i]][j])
            else:
                tx_v += ',' + str(val_sam[list(val_sam.keys())[i]][j])
    with open('train_sample.csv', 'w') as f:
        f.write(tx_t)
    with open('val_sample.csv', 'w') as f:
        f.write(tx_v)
    with open('test_sample.csv', 'w') as f:
        f.write(tx_)


def data_div_LSTM(root):
    df = csv(root)
    train = {}
    test = {}
    val = {}
    for i in df.keys():
        train[i] = []
        test[i] = []
        val[i] = []
    for j in range(len(df[df.keys()[0]])):
        for i in df.keys():
            if j < (4320 / 12 * 11):
                train[i].append(df[i][j])
            elif j < 4320:
                val[i].append(df[i][j])
            else:
                test[i].append(df[i][j])
    print(len(train['a']))
    test_sam = {}
    for i in train.keys():
        if i != 'PM2.5':
            test_sam[i] = []
    for i in range(1):
        test_sam['PM2.5_' + str(i + 1)] = []
    
    train_sam = {}
    for i in train.keys():
        if i != 'PM2.5':
            train_sam[i] = []
    for i in range(2):
        train_sam['PM2.5_' + str(i + 1)] = []
    
    val_sam = {}
    for i in train.keys():
        if i != 'PM2.5':
            val_sam[i] = []
    for i in range(2):
        val_sam['PM2.5_' + str(i + 1)] = []
    print(train['PM2.5'][0], val['PM2.5'][0])

    # 导入数据
    #train
    #print(len(train_sam['a']))
    for j in range(0, len(train['a']), 15 * 24):#每月15*24个样本点
        for k in range(1, 15 * 24):#每月前1个点不能做预测
            if j + k == 3960:
                print(j, k)
            for i in train.keys():
                if i != 'PM2.5':
                    train_sam[i].append(train[i][j + k - 1])
            for i in range(2):
                train_sam['PM2.5_' + str(i + 1)].append(train['PM2.5'][j + k - 1 + i])

    #val
    for k in range(1, 15 * 24):#每月前1个点不能做预测
        for i in val.keys():
            if i != 'PM2.5':
                val_sam[i].append(val[i][k - 1])
        for i in range(2):
            val_sam['PM2.5_' + str(i + 1)].append(val['PM2.5'][k - 1 + i])

    #test
    print(len(test_sam['a']))
    for j in range(0, len(test['a'])):
        for i in test.keys():
            if i != 'PM2.5':
                test_sam[i].append(test[i][j])
        for i in range(1):
            test_sam['PM2.5_' + str(i + 1)].append(test['PM2.5'][j + i])
    
    # 写入文件
    print(len(test_sam['a']))
    tx_ = ','.join(test_sam.keys())
    tx_t = tx_ + ',PM2.5_2'
    tx_v = tx_t
    for j in range(len(test_sam['a'])):
        tx_ += '\n'
        for i in range(len(test_sam.keys())):
            if i == 0:
                tx_ += str(test_sam[list(test_sam.keys())[i]][j])
            else:
                tx_ += ',' + str(test_sam[list(test_sam.keys())[i]][j])
    for j in range(len(train_sam['a'])):
        tx_t += '\n'
        for i in range(len(train_sam.keys())):
            if i == 0:
                tx_t += str(train_sam[list(train_sam.keys())[i]][j])
            else:
                tx_t += ',' + str(train_sam[list(train_sam.keys())[i]][j])
    for j in range(len(val_sam['a'])):
        tx_v += '\n'
        for i in range(len(val_sam.keys())):
            if i == 0:
                tx_v += str(val_sam[list(val_sam.keys())[i]][j])
            else:
                tx_v += ',' + str(val_sam[list(val_sam.keys())[i]][j])
    with open('train_sample_l.csv', 'w') as f:
        f.write(tx_t)
    with open('val_sample_l.csv', 'w') as f:
        f.write(tx_v)
    with open('test_sample_l.csv', 'w') as f:
        f.write(tx_)


if __name__ == '__main__':
    #df = csv('train.csv', usecols = range(3, 27), encoding = 'gbk')
    #df1 = csv('test1.csv', usecols = range(2, 11), encoding = 'gbk')
    #months, year = data_preprocessing(df, df1)
    #year, pm25_mx, pm25_mn = _guiyihua(year)
    data_div_LSTM(root = 'data_multi.csv')
