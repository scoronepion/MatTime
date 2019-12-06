# from __future__ import print_function
# import torch
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SimpleRNN, GRU, CuDNNLSTM, BatchNormalization
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
    df['Dmax'] += random.gauss(0, 0.06)
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = StandardScaler()
    scale_features = ['Tg', 'Tx', 'Tl', 'Dmax']
    df[scale_features] = scaler.fit_transform(df[scale_features])
    # frame as supervised learning
    reframed = series_to_supervised(df.loc[:, df.columns != 'Alloy'], 3, 1)
    # drop columns we don't want to predict
    reframed.drop(reframed.columns[[-2, -3, -4]], axis=1, inplace=True)
    print(reframed.head())
    return reframed

if __name__ == '__main__':
    raw = read_data().values
    # train_size = int(raw.shape[0] * 0.8)
    # train = torch.from_numpy(raw[:train_size, :])
    # test = torch.from_numpy(raw[train_size:, :])
    # train = raw[:train_size, :]
    # test = raw[train_size:, :]

    tscv = TimeSeriesSplit(n_splits=5)
    # keras model
    # design network
    for train_index, test_index in tscv.split(raw):

        train_x, train_y = raw[train_index, :-1], raw[train_index, -1]
        test_x, test_y = raw[test_index, :-1], raw[test_index, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
        test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
        print(train_x.shape, test_x.shape)

        model = Sequential()
        model.add(BatchNormalization())
        model.add(GRU(256, return_sequences=True))
        model.add(GRU(128, return_sequences=True))
        model.add(GRU(64))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        history = model.fit(train_x, train_y, epochs=7, batch_size=72, validation_data=(test_x, test_y), verbose=1, shuffle=False)

        yhat = model.predict(test_x)
        print(r2_score(test_y, yhat))

1 2 3 4 5
1, 2
1 2, 3
1 2 3, 4
1 2 3 4, 5