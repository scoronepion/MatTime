import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

def read_data():
    raw = pd.read_csv('NIMS Fatigue.csv').astype('float64')
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    raw['Fatigue'] = raw[['Fatigue']].apply(max_min_scaler)
    raw['Tensile'] = raw[['Tensile']].apply(max_min_scaler)
    raw['Fracture'] = raw[['Fracture']].apply(max_min_scaler)
    raw['Hardness'] = raw[['Hardness']].apply(max_min_scaler)
    return raw

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
        batch_size = 1

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

class MultiLossLayer(nn.Module):
    def __init__(self, list_length):
        super(MultiLossLayer, self).__init__()
        self._sigmas_sq = nn.ParameterList([nn.Parameter(torch.empty(())) for i in range(list_length)])
        for p in self.parameters():
            nn.init.uniform_(p, 0.2, 0.4)
        
    def forward(self, loss0, loss1, loss2, loss3):
        # loss0
        factor0 = torch.div(1.0, torch.mul(self._sigmas_sq[0], 2.0))
        loss = torch.add(torch.mul(factor0, loss0), 0.5 * torch.log(self._sigmas_sq[0]))
        # loss1
        factor1 = torch.div(1.0, torch.mul(self._sigmas_sq[1], 2.0))
        loss = torch.add(loss, torch.add(torch.mul(factor1, loss1), 0.5 * torch.log(self._sigmas_sq[1])))
        # loss2
        factor2 = torch.div(1.0, torch.mul(self._sigmas_sq[2], 2.0))
        loss = torch.add(loss, torch.add(torch.mul(factor2, loss2), 0.5 * torch.log(self._sigmas_sq[2])))
        # loss3
        factor3 = torch.div(1.0, torch.mul(self._sigmas_sq[3], 2.0))
        loss = torch.add(loss, torch.add(torch.mul(factor3, loss3), 0.5 * torch.log(self._sigmas_sq[3])))

        return loss

class cross_stitch(nn.Module):
    def __init__(self, length):
        '''
        length是两个input的最后一个维度和
        '''
        super(cross_stitch, self).__init__()
        self.matrix = nn.Parameter(torch.empty(length, length))
        nn.init.uniform_(self.matrix, 0.5, 0.8)

    def forward(self, input1, input2):
        # 这里数据除开batch，本来就是1维，应该不需要展平了
        input1_reshaped = torch.squeeze(input1)
        input2_reshaped = torch.squeeze(input2)
        input_reshaped = torch.cat([input1_reshaped, input2_reshaped], dim=-1)
        output = torch.matmul(input_reshaped, self.matrix)

        output1 = torch.reshape(output[:, :input1.size()[-1]], input1.size())
        output2 = torch.reshape(output[:, input2.size()[-1]:], input2.size())

        return output1, output2

class SingleModel(nn.Module):
    def __init__(self, length):
        super(SingleModel, self).__init__()
        self.attention1 = multi_heads_self_attention(feature_dim=length)
        self.attention2 = multi_heads_self_attention(feature_dim=length)
        self.linear = nn.Linear(length, 1)

    def forward(self, input):
        out, _ = self.attention1(input, input, input)
        out, _ = self.attention2(out, out, out)
        out = nn.functional.relu(self.linear(out))

        return out

class CrossModel(nn.Module):
    def __init__(self, length):
        super(CrossModel, self).__init__()
        # 因为有四个任务，所以需要有四路模型，中间三个十字绣单元
        # 每一路都采用两个attention模块，最后接一个线形层转化为最后的目标值
        self.attention1_1 = multi_heads_self_attention(feature_dim=length)
        self.attention1_2 = multi_heads_self_attention(feature_dim=length)
        self.linear1 = nn.Linear(length, 1)

        self.attention2_1 = multi_heads_self_attention(feature_dim=length)
        self.attention2_2 = multi_heads_self_attention(feature_dim=length)
        self.linear2 = nn.Linear(length, 1)

        self.attention3_1 = multi_heads_self_attention(feature_dim=length)
        self.attention3_2 = multi_heads_self_attention(feature_dim=length)
        self.linear3 = nn.Linear(length, 1)

        self.attention4_1 = multi_heads_self_attention(feature_dim=length)
        self.attention4_2 = multi_heads_self_attention(feature_dim=length)
        self.linear4 = nn.Linear(length, 1)

        self.cross_stitch1 = cross_stitch(length*2)
        self.cross_stitch2 = cross_stitch(length*2)

        self.loss1 = nn.MSELoss()
        self.loss2 = nn.MSELoss()
        self.loss3 = nn.MSELoss()
        self.loss4 = nn.MSELoss()

        self.lossLayer = MultiLossLayer(list_length=4)

    def forward(self, input, y1, y2, y3, y4):
        # y1=fatigue; y2=tensile; y3=fracture; y4=hardness;
        # 统一描述：out_路编号_阶段编号
        # 1 2路，第一阶段
        out1_1, _ = self.attention1_1(input, input, input)
        out2_1, _ = self.attention2_1(input, input, input)
        out1_1, out2_1 = self.cross_stitch1(out1_1, out2_1)
        # 1 2路，第二阶段
        out1_2, _ = self.attention1_2(out1_1, out1_1, out1_1)
        out2_2, _ = self.attention2_2(out2_1, out2_1, out2_1)
        # 1 2路，第三阶段
        out1_3 = nn.functional.relu(self.linear1(out1_2))
        out2_3 = nn.functional.relu(self.linear2(out2_2))

        # 3 4路，第一阶段
        out3_1, _ = self.attention3_1(input, input, input)
        out4_1, _ = self.attention4_1(input, input, input)
        out3_1, out4_1 = self.cross_stitch2(out3_1, out4_1)
        # 3 4路，第二阶段
        out3_2, _ = self.attention3_2(out3_1, out3_1, out3_1)
        out4_2, _ = self.attention4_2(out4_1, out4_1, out4_1)
        # 3 4路，第三阶段
        out3_3 = nn.functional.relu(self.linear3(out3_2))
        out4_3 = nn.functional.relu(self.linear4(out4_2))

        # 分任务计算损失
        loss1 = self.loss1(torch.squeeze(out1_3), torch.squeeze(y1))
        loss2 = self.loss2(torch.squeeze(out2_3), torch.squeeze(y2))
        loss3 = self.loss3(torch.squeeze(out3_3), torch.squeeze(y3))
        loss4 = self.loss4(torch.squeeze(out4_3), torch.squeeze(y4))

        loss = self.lossLayer(loss1, loss2, loss3, loss4)

        return loss, out1_3, out2_3, out3_3, out4_3

if __name__ =='__main__':
    raw = read_data().values
    print(raw.shape)
    features = raw[:, :16]
    targets = raw[:, 16:]

    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.4)

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)

    fatigue_train = torch.from_numpy(y_train[:, 0]).view(-1, 1)
    fatigue_test = torch.from_numpy(y_test[:, 0]).view(-1, 1)

    tensile_train = torch.from_numpy(y_train[:, 1]).view(-1, 1)
    tensile_test = torch.from_numpy(y_test[:, 1]).view(-1, 1)

    fracture_train = torch.from_numpy(y_train[:, 2]).view(-1, 1)
    fracture_test = torch.from_numpy(y_test[:, 2]).view(-1, 1)

    hardness_train = torch.from_numpy(y_train[:, 3]).view(-1, 1)
    hardness_test = torch.from_numpy(y_test[:, 3]).view(-1, 1)

    model = SingleModel(length=x_train.size()[1])
    model.double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    y_train, y_test = tensile_train, tensile_test

    epoch_num = 50000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            out = model(x_train)
            loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
            loss.backward()

            return loss

        optimizer.step(closure)

        if epoch % 10 == 9:
            print('epoch: {}'.format(epoch))

            pred = model(x_test)
            loss = criterion(torch.squeeze(pred), torch.squeeze(y_test))
            r2 = r2_score(torch.squeeze(pred).detach().numpy(), torch.squeeze(y_test).detach().numpy())
            print('loss={}, r2={}'.format(loss, r2))