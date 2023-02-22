要求：
环境搭建
"""
python -m pip install pandas, numpy, torch, matplotlib
"""

模型：

1.线性回归模型
运行模型
"""
python task1_pm25.py
"""
这个命令包括在给定的训练集上训练和验证模型，并在给定的测试集上进行预测。结果将被存储在 "sub.csv "中。

2.LSTM
数据预处理
"""
python data_management.py
"""
这个命令将帮助把数据切割成数据帧。

运行模型
"""
python LSTM_based.py
"""
这个命令包括在数据集上训练、验证和测试模型，并对原始预测数据集进行预测。结果将存储在'LSTM_pred.csv'中。


数据集：

给定训练集：train.csv (encoding='gbk')
给定测试集：test1.csv (encoding='gbk')

LSTM训练集：train_sample_l.csv
LSTM验证集：val_sample_l.csv
LSTM测试集：val1_sample_l.csv
LSTM预测数据集：test_sample_l.csv


结果：

我的训练结果在sample_submission.csv中。