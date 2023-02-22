Requirement:
building environment
"""
python -m pip install pandas, numpy, torch, matplotlib
"""

Models:

1.Linear Regression Model
Running model
"""
python task1_pm25.py
"""
This command includes training and validation of the model on the given train set and doing a prediction on the given test set. The result will be stored in 'sub.csv'

2.LSTM
Data Preprocessing
"""
python data_management.py
"""
This command will help cutting data into dataframes\

Running model
"""
python LSTM_based.py
"""
This command includes training and validation and testing of the model on the dataset and doing a prediction on the original prediction data set. The result will be stored in 'LSTM_pred.csv'


Dataset:

given train set: train.csv (encoding='gbk')
given test set: test1.csv (encoding='gbk')

LSTM train set: train_sample_l.csv
LSTM validation set: val_sample_l.csv
LSTM test set: val1_sample_l.csv
LSTM prediction data set: test_sample_l.csv


Result:

The result of my training is in sample_submission.csv