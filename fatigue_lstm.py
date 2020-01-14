import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import random
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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

def read_data(reframe=True, platform=None):
    if platform == 'v100':
        df = pd.read_csv('/home/lab106/zy/MatTime/fatigue.csv')
    else:
        df = pd.read_csv('fatigue.csv')
    df.drop(columns=['dA', 'dB', 'dC'], inplace=True)
    # 重排数据
    df.sort_values("Fatigue", inplace=True)
    df = df.reset_index(drop=True)
    # features = ['RedRatio', 'dA', 'dB', 'dC', 'Fatigue']
    features = ['RedRatio', 'Fatigue']
    # df[features] += random.gauss(0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    if reframe:
        # frame as supervised learning
        # reframed = series_to_supervised(df.loc[:, df.columns != 'Alloy'], 3, 1)
        reframed = series_to_supervised(df, 3, 1)
    else:
        # reframed = df.loc[:, df.columns != 'Alloy'].dropna()
        reframed = df.dropna()

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

class multi_heads_self_attention(nn.Module):
    def __init__(self, feature_dim=56, num_heads=2, dropout=0.0):
        super(multi_heads_self_attention, self).__init__()

        self.dim_per_head = feature_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(feature_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(feature_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(feature_dim, self.dim_per_head * num_heads)

        self.sdp_attention = scaled_dot_product_attention(dropout)
        self.linear_attention = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)
        # self.linear_1 = nn.Linear(feature_dim, 256)
        # self.linear_2 = nn.Linear(256, feature_dim)
        # self.layer_final = nn.Linear(feature_dim, 3)

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

        output = self.linear_attention(context)
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        # # pass through linear
        # output = nn.functional.relu(self.linear_1(output))
        # output = nn.functional.relu(self.linear_2(output))

        # # pass through layer final
        # output = self.layer_final(output)

        return output, attention

class AT_LSTM(nn.Module):
    '''
    LSTM 的输出送入 Attention 中，随后再与线性层连接，最后得到预测的值
    '''
    def __init__(self, feature_dim, hidden_size, head_num):
        super(AT_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, num_layers=2)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.attention = multi_heads_self_attention(feature_dim=hidden_size, num_heads=head_num)
        # self.layer_trans = nn.Linear(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, input):
        output, _ = self.lstm(input) 
        output = self.layer_norm(output)
        output, _ = self.attention(output, output, output)
        # output = nn.functional.relu(self.layer_trans(output))
        output = self.linear(output)

        return output

if __name__ == '__main__':
    # f = open('0114.log', 'a')
    # sys.stdout = f
    # sys.stderr = f
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    print(read_data(reframe=True).head())
    print(read_data(reframe=True).info())
    raw = read_data(reframe=True).values
    raw = np.expand_dims(raw, axis=1)
    train_size = int(raw.shape[0] * 0.8)
    # print(raw.info())
    print(raw.shape)

    if torch.cuda.is_available():
        train = torch.from_numpy(raw[:train_size, :, :]).to(device)
        test = torch.from_numpy(raw[train_size:, :, :]).to(device)
    else:
        train = torch.from_numpy(raw[:train_size, :, :])
        test = torch.from_numpy(raw[train_size:, :, :])
    train_x, train_y = train[:, :, :-1], train[:, :, -1]
    test_x, test_y = test[:, :, :-1], test[:, :, -1]
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

    at_lstm = AT_LSTM(feature_dim=train.size()[-1]-1, hidden_size=256, head_num=1)
    if torch.cuda.is_available():
        at_lstm.to(device)
    at_lstm.double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(at_lstm.parameters(), lr=0.001)

    epoch_num = 10000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            out = at_lstm(train_x)
            loss = criterion(torch.squeeze(out), torch.squeeze(train_y))
            loss.backward()
            return loss

        optimizer.step(closure)

        if epoch % 10 == 9:
            print("Epoch : ", epoch)
            pred = at_lstm(test_x)
            loss = criterion(torch.squeeze(pred), torch.squeeze(test_y))
            print("test loss : ", loss.data.item())
            print("r2 : ", r2_score(torch.squeeze(test_y).detach().numpy(), torch.squeeze(pred).detach().numpy()))

    # f.close()