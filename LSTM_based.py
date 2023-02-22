import torch
from torch import nn
import pandas as pd
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)  # set initial datatype of tensor
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# data pre process
def dataProcess(root: str = '', usage: str = '', model_name: str = ''):
    usages_ = ['train', 'val', 'test']
    if usage not in usages_:
        raise RuntimeError('Usage isn`t right.')

    names_ = ['single', 'multi', 'LSTM']
    if model_name not in names_:
        raise RuntimeError('model name isn`t right.')

    if root == '':
        raise RuntimeError('Root is invalide. Shouldn`t be an empty string.')

    test_ = False
    if usage == 'test':
        test_ = True

    if model_name == 'single':
        usecols_ = range(5, 6)
    elif model_name == 'multi':
        usecols_ = range(0, 6)
    else:
        usecols_ = [0, 1, 2, 3, 4, 5]
    if not test_:
        usecols_.append(6)
    df = pd.read_csv(root, usecols=usecols_, dtype=np.float32)
    all_cols = ['a', 'b', 'c', 'd', 'e', 'PM2.5_1']
    training_cols = []
    for i in df.columns:
        if str(i) in all_cols:
            training_cols.append(str(i))
    if not test_:
        y_list = np.array(df['PM2.5_2'].values).T

    x_list = np.array([df[i].values for i in training_cols]).T

    x = np.array(x_list)
    if not test_:
        y = np.array(y_list)
    else:
        y = np.array(np.zeros(len(x)))
    return x, y


# data reshaped into seqlength
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction

    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape
    months = num_samples / 359
    x_n = np.zeros((int(months * (359 - seq_length + 1)), seq_length, num_features))
    y_n = np.zeros((int(months * (359 - seq_length + 1)), 1))
    j = 0
    for i in range(x_n.shape[0]):
        if int((i + seq_length + 1) / 359) == int((i + 1) / 359):
            x_n[j, :, :num_features] = x[i:i + seq_length, :]
            y_n[j, :] = y[i + seq_length - 1]
            j += 1
    x_b = []
    y_b = []
    for i in range(0, int(len(x_n)), batch_size):
        l = batch_size
        if i + batch_size >= len(x_n):
            l = len(x_n) - i
        x_b.append(x_n[i:i + l, :, :])
        y_b.append(y_n[i:i + l, :])

    return x_b, y_b


# testing data reshaped into seqlength
def reshape_data_test(xt: np.ndarray, yt: np.ndarray, seq_length: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction

    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = xt.shape
    x_nt = np.zeros((int(num_samples / seq_length), seq_length, num_features))
    y_nt = np.zeros((int(num_samples / seq_length), 1))

    for i in range(0, num_samples, seq_length):
        x_nt[int(i / seq_length), :, :num_features] = xt[i:i + seq_length, :]
        y_nt[int(i / seq_length), :] = yt[i + seq_length - 1]
    x_bt = [x_nt[:, :, :], ]
    y_bt = [y_nt[:, :], ]

    return x_bt, y_bt


# calculating L1
def L1_(x, y, w, b):
    l1 = 0
    # print(len(x))
    for i in range(len(x)):
        l1 += abs_(y[i] - (w.dot(x[i]) + b))
    return l1 / len(x)


# calculating L2
def L2_(x, y, w, b):
    l2 = 0
    for i in range(len(x)):
        l2 += (y[i] - (w.dot(x[i]) + b)) ** 2
    return l2 / len(x)


# calculation abs
def abs_(ori: float = 0):
    return (ori if ori > 0.0 else -ori)


# training linear regression
def train_lr(x_train, y_train, epoch):
    bias = 0  # bias init
    weights = np.ones(14)  # weight init
    # weights = np.random.rand(14) # weight randomly init
    learning_rate = 1  # lr init
    reg_rate = 0.001  # rr init
    bg2_sum = 0  # Σ bias ** 2
    wg2_sum = np.zeros(14)  # Σ weight ** 2

    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(14)
        # calculating g of Loss_label on every sample
        for j in range(len(x_train)):
            b_g += (y_train[j] - weights.dot(x_train[j, 1, :]) - bias) * (-1)
            for k in range(9):
                w_g[k] += (y_train[j] - weights.dot(x_train[j, 1, :]) - bias) * (-x_train[j, 1, k])
        # average
        b_g /= len(x_train)
        w_g /= len(x_train)

        # g on w of Loss_regularization
        for m in range(9):
            w_g[m] += reg_rate * weights[m]

        # adding g
        bg2_sum += b_g ** 2
        wg2_sum += w_g ** 2
        # updating w and b
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g

        # loss(100 epochs/printing)
        if i % 100 == 0:
            loss = L1_(x_train[:, 1], y_train, weights, bias)
            print('epoch={}, loss='.format(i), loss)

    return weights, bias


# prediction of linear regression model
def pred_lr(x_val, weights, bias):
    y_val = np.zeros(len(x_val))
    for j in range(len(x_val)):
        y_pred = weights.dot(x_val[j, 1, :]) + bias
        y_val[j] = y_pred
    tx = ''
    for i in range(len(y_val)):
        tx += str(y_val[i]) + '\n'
        # print(y_val[i])
    with open('sub.csv', 'w') as f:
        f.write(tx)
    return y_val


# training LSTM model
def train_(model, optimizer, datas, loss_func):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param datas: A Tuple, providing the trainings data in mini batches.
    :param loss_func: The loss function to minimize.
    """
    # set model to train mode (important for dropout)
    model.train()
    # request mini-batch of data from the loader
    xs, ys = datas
    # print(len(xs))
    for i in range(len(xs)):
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        x_, y_ = torch.tensor(xs[i]).to(DEVICE), torch.tensor(ys[i]).to(DEVICE)
        # print(x_.size())
        # get model predictions
        y_hat = model(x_)
        # calculate loss
        loss = loss_func(y_hat, y_)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
    return loss.item()


# validation LSTM model
def eval_(model, datas):
    model.eval()
    obs = []
    pred = []
    with torch.no_grad():
        xs, ys = datas
        for i in range(len(xs)):
            x = torch.tensor(xs[i]).to(DEVICE)
            y = torch.tensor(ys[i]).to(DEVICE)
            y_p = model(x)
            obs.append(y)
            # print(y)
            pred.append(y_p)
    # print(obs, pred)
    return torch.cat(obs), torch.cat(pred)


# test LSTM model
def test_(model, datas):
    model.eval()
    pred = []
    with torch.no_grad():
        xs, ys = datas
        for i in range(len(xs)):
            x = torch.tensor(xs[i]).to(DEVICE)
            y_p = model(x)
            pred.append(y_p)
    return torch.cat(pred)


class Model(nn.Module):
    def __init__(self, hidden_size: int, dropout_rate: float = 0.01):
        """Initialize model

        :param hidden_size: Number of hidden units/LSTM cells
        :param dropout_rate: Dropout rate of the last fully connected
            layer. Default 0.0
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate

        # create required layer
        ip_size = 6
        self.lstm = nn.LSTM(input_size=ip_size, hidden_size=self.hidden_size,
                            num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Network.

        :param x: Tensor of shape [batch size, seq length, num features]
            containing the input data for the LSTM network.

        :return: Tensor containing the network predictions
        """
        # print(x.size())
        # x = torch.tensor(x, dtype=torch.float32)
        output, (h_n, c_n) = self.lstm(x)

        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1, :, :]))
        return pred


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # using L2 loss
        return torch.mean(torch.pow((y_pred - y_true), 2))


def lr_reduce(epoch, lr):
    a = 1
    if epoch > 220:
        a = 0.002
    return lr * a


def calc_val(obs: np.array, sim: np.array):
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator
    return nse_val


def LSTM_main():
    # recording losses
    loss_ = {'l1': [], 'l2': []}
    nses = []

    # basic model setings
    sequence_length = 9  # int((24 * 15 - 9) / 9)
    learning_rate = 1e-3
    dropout_rate = 0.0
    hidden_size = 15
    batch_size = 512
    epochs = 300

    # model defining
    model = Model(hidden_size=hidden_size, dropout_rate=dropout_rate).to(DEVICE)
    loss_func = My_loss()

    # loading datas
    x_tr, y_tr = dataProcess('train_sample_l.csv', 'train', 'LSTM')
    x_tr, y_tr = reshape_data(x_tr, y_tr, sequence_length, batch_size)
    x_val, y_val = dataProcess('val_sample_l.csv', 'val', 'LSTM')
    x_val, y_val = reshape_data(x_val, y_val, sequence_length, batch_size)

    df = pd.read_csv('data_gscale.txt')
    mul_, plus_ = float(df[df.columns[0]][0]), float(df[df.columns[1]][0])
    # print(mul_, plus_)

    for i in range(epochs):
        lr=lr_reduce(i + 1, learning_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss = train_(model, optimizer, (x_tr, y_tr), loss_func)
        obs, preds = eval_(model, (x_val, y_val))
        obs = obs.cpu().numpy()
        preds = preds.cpu().numpy()
        if(i + 1) % 20 == 0:
            l1 = L1_(preds, obs, np.array(1, ), 0)  # calculating L1 loss
            loss_['l1'].append(str(float(l1[0]) * mul_))
            loss_['l2'].append(str(loss * mul_ * mul_))
            nse = calc_val(obs, preds)
            nses.append(nse)
            print(f'epoch = {i + 1}\nlr is {lr}\nL1 loss is {float(l1[0]) * mul_}\nL2 loss is {loss * mul_ * mul_}\nNSE is {nse}')

    # testing model
    x_val1, y_val1 = dataProcess('val1_sample_l.csv', 'val', 'LSTM')
    x_val1, y_val1 = reshape_data(x_val1, y_val1, sequence_length, batch_size)
    obs, preds = eval_(model, (x_val1, y_val1))
    obs = obs.cpu().numpy()
    preds = preds.cpu().numpy()
    l1 = L1_(preds, obs, np.array(1, ), 0)  # calculating L1 loss
    print(len(l1))
    loss_['l1'].append(str(float(l1) * mul_))
    loss_['l2'].append(str(loss * mul_ * mul_))
    nse = calc_val(obs, preds)
    print(f'Testing:\nL1 loss is {float(l1[0]) * mul_}\nL2 loss is {loss * mul_ * mul_}\nNSE is {nse}')

    # ploting
    _, ax1 = plt.subplots(figsize=(12, 4))
    data_range = [str(i + 1) for i in range(len(nses))]
    ax1.plot(data_range, nses, label=f"NSE")
    ax1.legend()
    ax1.set_title(f"LSTM,epoch={epochs}")
    ax1.set_xlabel("epochs")
    ax1.set_xticks([])
    _ = ax1.set_ylabel("NSE")

    _, ax = plt.subplots(figsize=(12, 4))
    data_range = [str(i + 1) for i in range(len(obs))]
    ax.plot(data_range, obs, label=f"observation, NSE = {nse:.3f}")
    ax.plot(data_range, preds, label=f"prediction of {epochs} epochs")
    ax.legend()
    ax.set_title(f"LSTM,epoch={epochs}")
    ax.set_xlabel("Date")
    ax.set_xticks([])
    _ = ax.set_ylabel("PM2.5")

    # running model on given test set
    x, y = dataProcess('test_sample_l.csv', 'test', 'LSTM')
    x, y = reshape_data_test(x, y, sequence_length)
    f = pd.read_csv('data_gscale.txt')
    pred = test_(model, (x, y))
    pred = pred.cpu().numpy()
    PM25 = []
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            pred[i][j] *= mul_
            pred[i][j] += plus_
            if pred[i][j] <= -1:
                pred[i][j] = -1
            elif pred[i][j] < 0:
                pred[i][j] = 0
            else:
                pred[i][j] = int(pred[i][j] + 0.5)
            PM25.append(str(pred[i][j]))
    print('total sample =', len(PM25))
    tx = ''
    tx = '\n'.join(list(PM25))
    tx += '\n'
    tx += ','.join(loss_['l1'])
    tx += '\n'
    tx += ','.join(loss_['l2'])
    tx += '\n'
    tx += str(loss_['l1'][-1]) + ',' + str(loss_['l2'][-1])
    with open('LSTM_pred.csv', 'w') as f:
        f.write(tx)
    plt.show()


if __name__ == '__main__':
    LSTM_main()
