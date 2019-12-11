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

def read_data(reframe=True):
    df = pd.read_csv('/home/lab106/zy/MatTime/ctt.csv')
    # 重排数据
    df.sort_values("Dmax", inplace=True)
    df = df.reset_index(drop=True)
    df.drop(df.index[[-1]], inplace=True)
    features = ['Tg', 'Tx', 'Tl', 'Dmax']
    df[features] += random.gauss(0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    if reframe:
        # frame as supervised learning
        reframed = series_to_supervised(df.loc[:, df.columns != 'Alloy'], 3, 1)
        # drop columns we don't want to predict
        reframed.drop(reframed.columns[[-2, -3, -4]], axis=1, inplace=True)
    else:
        reframed = df.loc[:, df.columns != 'Alloy'].dropna()

    # print(reframed.head())
    return reframed

class scaled_dot_product_attention(nn.Module):
    def __init__(self, att_dropout=0.0):
        super(scaled_dot_product_attention, self).__init__()
        self.dropout = nn.Dropout(att_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None):
        '''
        args:
            q: [batch_size, q_length, q_dimension]
            k: [batch_size, k_length, k_dimension]
            v: [batch_size, v_length, v_dimension]
            q_dimension = k_dimension = v_dimension
            scale: 缩放因子
        return:
            context, attention
        '''
        # 快使用神奇的爱因斯坦求和约定吧！
        attention = torch.einsum('ijk,ilk->ijl', [q, k])
        if scale:
            attention = attention * scale
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.einsum('ijl,ilk->ijk', [attention, v])
        return context, attention

# TODO:调试
class multi_heads_self_attention(nn.Module):
    def __init__(self, feature_dim=64, num_heads=4, dropout=0.0):
        super(multi_heads_self_attention, self).__init__()

        self.dim_per_head = feature_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(feature_dim, feature_dim)
        self.linear_k = nn.Linear(feature_dim, feature_dim)
        self.linear_v = nn.Linear(feature_dim, feature_dim)

        self.sdp_attention = scaled_dot_product_attention(dropout)
        self.linear_final = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, key, value, query):
        residual = query
        batch_size = key.size(0)

        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * self.num_heads, -1, self.dim_per_head)
        value = value.view(batch_size * self.num_heads, -1, self.dim_per_head)
        query = query.view(batch_size * self.num_heads, -1, self.dim_per_head)

        scale = (key.size(-1) // self.num_heads) ** -0.5
        context, attention = self.sdp_attention(query, key, value, scale)

        # concat heads
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)

        output = self.linear_final(context)
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention

class Sequence(nn.Module):
    def __init__(self, feature_num):
        super(Sequence, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_num, hidden_size=256, num_layers=3)
        self.gru = nn.GRU(input_size=feature_num, hidden_size=256, num_layers=3)
        self.rnn = nn.RNN(input_size=feature_num, hidden_size=256, num_layers=3)
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

    tscv = TimeSeriesSplit(n_splits=5)

    for train_index, test_index in tscv.split(raw):
        # data processing
        train = torch.from_numpy(raw[train_index, :]).to(device)
        test = torch.from_numpy(raw[test_index, :]).to(device)
        train_x, train_y = train[:, :-1], train[:, -1]
        test_x, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        # train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
        # test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
        train_x = train_x.reshape((-1, 2, train_x.shape[1]))
        test_x = test_x.reshape((-1, 2, test_x.shape[1]))
        print(train_x.shape, test_x.shape)

        # build the model
        seq = Sequence(feature_num=12)
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