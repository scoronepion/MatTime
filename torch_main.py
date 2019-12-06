import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
import random
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# convert series to supervised learning
def series_to_supervised(df, n_in=1, n_out=1, dropnan=True):
	n_vars =  df.shape[1]
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def read_data():
    df = pd.read_csv('/home/lab106/zy/CTT/ctt.csv')
    # 重排数据
    df.sort_values("Dmax", inplace=True)
    df = df.reset_index(drop=True)
    df.drop(df.index[[-1]], inplace=True)
    features = ['Tg', 'Tx', 'Tl', 'Dmax']
    df[features] += random.gauss(0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    # frame as supervised learning
    reframed = series_to_supervised(df.loc[:, df.columns != 'Alloy'], 3, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[-2, -3, -4]], axis=1, inplace=True)
    # reframed = df.loc[:, df.columns != 'Alloy'].dropna()
    # print(reframed.head())
    return reframed

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTM(input_size=12, hidden_size=256, num_layers=3)
        self.gru = nn.GRU(input_size=12, hidden_size=256, num_layers=3)
        self.rnn = nn.RNN(input_size=3, hidden_size=256, num_layers=3)
        self.linear = nn.Linear(256, 1)

    def forward(self, input, future = 0):
        outputs = []

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            output, (_, _) = self.lstm(input_t)
            # output, _ = self.gru(input_t)
            # output, _ = self.rnn(input_t)
            output = self.linear(output)
            outputs += [output]
        # for i in range(future):# if we should predict the future
        #     h_t, c_t = self.lstm1(output, (h_t, c_t))
        #     h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
        #     output = self.linear(h_t2)
        #     outputs += [output]
        outputs = torch.stack(outputs, 1).squeeze()
        return outputs

if __name__ == '__main__':
    raw = read_data().values
    train_size = int(raw.shape[0] * 0.8)

    device = torch.device('cuda:0')

    # train = torch.from_numpy(raw[:train_size, :]).to(device)
    # test = torch.from_numpy(raw[train_size:, :]).to(device)
    # train_x, train_y = train[:, :-1], train[:, -1]
    # test_x, test_y = test[:, :-1], test[:, -1]
    # # reshape input to be 3D [samples, timesteps, features]
    # train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    # test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    # print(train_x.shape, test_x.shape)

    tscv = TimeSeriesSplit(n_splits=8)

    for train_index, test_index in tscv.split(raw):
        # data processing
        train = torch.from_numpy(raw[train_index, :]).to(device)
        test = torch.from_numpy(raw[test_index, :]).to(device)
        train_x, train_y = train[:, :-1], train[:, -1]
        test_x, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
        test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
        print(train_x.shape, test_x.shape)

        # build the model
        seq = Sequence()
        seq.to(device)
        seq.double()
        criterion = nn.MSELoss()
        # use LBFGS as optimizer since we can load the whole data to train
        # optimizer = optim.LBFGS(seq.parameters(), tolerance_grad=1e-4)
        optimizer = optim.Adam(seq.parameters())

        loss_list = []
        test_loss_list = []
        epoch_num = 1000

        # begin to train
        for epoch in range(epoch_num):

            def closure():
                optimizer.zero_grad()
                out = seq(train_x)
                loss = criterion(out, train_y)
                # print('loss:', loss.data.numpy())
                loss_list.append(loss.data.item())
                loss.backward()
                return loss

            optimizer.step(closure)

            if epoch % 100 == 99:
                print('epoch : ', epoch)
                pred = seq(test_x)
                loss = criterion(pred, test_y)
                print('test loss:', loss.data.item())
                print('r2:', r2_score(test_y.cpu(), pred.data.cpu()))