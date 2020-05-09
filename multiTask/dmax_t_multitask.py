import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import math
# import imblearn
from MTL_predata import mtl_composition, mtl_atomic, composition_atomic_feature
# from imblearn.over_sampling import SMOTE
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, classification_report

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
    def __init__(self, feature_dim, num_heads=1, dropout=0.0):
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

        if key.size(-1) // self.num_heads != 0:
            scale = (key.size(-1) // self.num_heads) ** -0.5
        else:
            scale = 1
        context, attention = self.sdp_attention(query, key, value, scale)

        # concat heads
        context = context.view(batch_size, -1, self.dim_per_head * self.num_heads)

        output = self.linear_attention(context)
        output = self.dropout(output)
        # output = torch.squeeze(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        # # pass through linear
        # output = nn.functional.relu(self.linear_1(output))
        # output = nn.functional.relu(self.linear_2(output))

        # # pass through layer final
        # output = self.layer_final(output)

        return output, attention

class embedding_attention(nn.Module):
    def __init__(self, feature_dim):
        super(embedding_attention, self).__init__()
        self.attention = multi_heads_self_attention(feature_dim=feature_dim)
        self.linear_final = nn.Linear(feature_dim, 1)

    def forward(self, input):
        output, _ = self.attention(input, input, input)
        output = self.linear_final(output)

        return output

class MultiLossLayer(nn.Module):
    def __init__(self, list_length):
        super(MultiLossLayer, self).__init__()
        self._sigmas_sq = nn.ParameterList([nn.Parameter(torch.empty(())) for i in range(list_length)])
        for p in self.parameters():
            nn.init.uniform_(p,0.2,1.0)
        
    def forward(self, regression_loss, classifier_loss):
        # regression loss
        factor0 = torch.div(1.0,torch.mul(self._sigmas_sq[0], 2.0))
        loss = torch.add(torch.mul(factor0, regression_loss), 0.5 * torch.log(self._sigmas_sq[0]))
        # classification loss
        factor1 = torch.div(1.0,torch.mul(self._sigmas_sq[1], 1.0))
        loss = torch.add(loss, torch.add(torch.mul(factor1,classifier_loss), 0.5 * torch.log(self._sigmas_sq[1])))

        return loss

class multitask(nn.Module):
    def __init__(self, feature_dim):
        super(multitask, self).__init__()
        self.dmax_reg = embedding_attention(feature_dim=feature_dim)
        self.t_reg = embedding_attention(feature_dim=feature_dim)
        self.Layer_multi_loss = MultiLossLayer(2)
        self.dmax_reg_loss = nn.MSELoss()
        self.t_reg_loss = nn.MSELoss()

    def forward(self, x_dmax, y_dmax, x_t, y_t):
        out_dmax = self.dmax_reg(x_dmax)
        out_t = self.t_reg(x_t)
        loss_dmax = self.dmax_reg_loss(torch.squeeze(out_dmax), torch.squeeze(y_dmax))
        loss_t = self.t_reg_loss(torch.squeeze(out_t), torch.squeeze(y_t))
        multi_loss = self.Layer_multi_loss(loss_dmax, loss_t)

        return multi_loss, loss_dmax, loss_t, out_dmax, out_t

if __name__ == '__main__':
    raw = composition_atomic_feature()
    features = raw.iloc[:, :-2].values
    target_dmax = raw.iloc[:, -2:-1].values
    target_t = raw.iloc[:, -1:].values

    x_train_dmax, x_test_dmax, y_train_dmax, y_test_dmax = train_test_split(features, target_dmax, test_size=0.1)
    x_train_t, x_test_t, y_train_t, y_test_t = train_test_split(features, target_t, test_size=0.1)
    batch_size = 1  
    features_size = features.shape[1]
    # print(features.shape)
    # print(x_train_t.shape)
    # target_size = 3

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        # dmax task
        x_train_dmax = torch.from_numpy(x_train_dmax).view(batch_size, -1, features_size).to(device)
        y_train_dmax = torch.from_numpy(y_train_dmax).view(batch_size, -1).to(device)

        x_test_dmax = torch.from_numpy(x_test_dmax).view(batch_size, -1, features_size).to(device)
        y_test_dmax = torch.from_numpy(y_test_dmax).view(batch_size, -1).to(device)

        # t task
        x_train_t = torch.from_numpy(x_train_t).view(batch_size, -1, features_size).to(device)
        y_train_t = torch.from_numpy(y_train_t).view(batch_size, -1).to(device)

        x_test_t = torch.from_numpy(x_test_t).view(batch_size, -1, features_size).to(device)
        y_test_t = torch.from_numpy(y_test_t).view(batch_size, -1).to(device)
    else:
        # dmax task
        x_train_dmax = torch.from_numpy(x_train_dmax).view(batch_size, -1, features_size)
        y_train_dmax = torch.from_numpy(y_train_dmax).view(batch_size, -1)

        x_test_dmax = torch.from_numpy(x_test_dmax).view(batch_size, -1, features_size)
        y_test_dmax = torch.from_numpy(y_test_dmax).view(batch_size, -1)

        # t task
        x_train_t = torch.from_numpy(x_train_t).view(batch_size, -1, features_size)
        y_train_t = torch.from_numpy(y_train_t).view(batch_size, -1)

        x_test_t = torch.from_numpy(x_test_t).view(batch_size, -1, features_size)
        y_test_t = torch.from_numpy(y_test_t).view(batch_size, -1)

    model = multitask(feature_dim=features_size)
    if torch.cuda.is_available():
        model.to(device)
    model.double()

    # criterion = nn.CrossEntropyLoss()
    params = [
        {'params': model.Layer_multi_loss.parameters(), 'lr': 0.0001},
        {'params': model.dmax_reg.parameters(), 'lr': 0.0001},
        {'params': model.t_reg.parameters(), 'lr': 0.001}
    ]
    optimizer = optim.Adam(params)

    epoch_num = 100000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            loss, _, _, _, _ = model(x_train_dmax, y_train_dmax, x_train_t, y_train_t)
            # loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
            # print('loss:', loss.data.item())
            # loss_list.append(loss.data.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        if epoch % 100 == 99:
            print('epoch : ', epoch)
            loss, loss_dmax, loss_t, output_dmax, output_t = model(x_test_dmax, y_test_dmax, x_test_t, y_test_t)
            print('test multi loss:', loss.data.item())
            # print('test classification loss:', loss_dmax.data.item())
            # print('test regression loss:', loss_t.data.item())

            print('dmax r2:', r2_score(torch.squeeze(y_test_dmax.cpu()), torch.squeeze(output_dmax.data.cpu())))
            print('t r2:', r2_score(torch.squeeze(y_test_t.cpu()), torch.squeeze(output_t.data.cpu())))