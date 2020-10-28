import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. 单任务网络结果
# 2. 多任务去除十字绣结果
# 3. 多任务去除加权损失函数结果
# 4. 多任务去除attention结果

def read_nims_data():
    raw = pd.read_csv('/home/lab106/zy/new_geogre_data/NIMS Fatigue.csv').astype('float64')
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    raw['Fatigue'] = raw[['Fatigue']].apply(max_min_scaler)
    raw['Tensile'] = raw[['Tensile']].apply(max_min_scaler)
    raw['Fracture'] = raw[['Fracture']].apply(max_min_scaler)
    raw['Hardness'] = raw[['Hardness']].apply(max_min_scaler)
    return raw

def read_dmax_data():
    raw = pd.read_csv('/home/lab106/zy/new_geogre_data/fulldmaxttt.csv').dropna()
    raw.drop(columns=['Phase Formation', 'Alloy Formula'], inplace=True)
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))
    gauss_scaler = lambda x: x + np.abs(random.gauss(0, 0.01))
    raw['Dmax'] = raw[['Dmax']].apply(max_min_scaler).apply(gauss_scaler)
    raw['Tg'] = raw[['Tg']].apply(max_min_scaler)
    raw['Tx'] = raw[['Tx']].apply(max_min_scaler)
    raw['Tl'] = raw[['Tl']].apply(max_min_scaler)
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
            nn.init.uniform_(p, 0.5, 0.8)
        
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
        nn.init.uniform_(self.matrix, 0.3, 0.5)

    def forward(self, input1, input2):
        # 这里数据除开batch，本来就是1维，应该不需要展平了
        input1_reshaped = torch.squeeze(input1)
        input2_reshaped = torch.squeeze(input2)
        input_reshaped = torch.cat([input1_reshaped, input2_reshaped], dim=-1)
        output = torch.matmul(input_reshaped, self.matrix)

        output1 = torch.reshape(output[:, :input1.size()[-1]], input1.size())
        output2 = torch.reshape(output[:, input2.size()[-1]:], input2.size())

        return output1, output2

# 1. 单任务网络结果
class model1(nn.Module):
    def __init__(self, length):
        super(model1, self).__init__()
        self.attention1 = multi_heads_self_attention(feature_dim=length)
        self.attention2 = multi_heads_self_attention(feature_dim=length)
        self.linear = nn.Linear(length, 1)

    def forward(self, input):
        out, _ = self.attention1(input, input, input)
        out, _ = self.attention2(out, out, out)
        out = nn.functional.relu(self.linear(out))

        return out

# 2. 多任务去除十字绣结果
class model2(nn.Module):
    def __init__(self, length):
        super(model2, self).__init__()
        # 因为有四个任务，所以需要有四路模型，中间三个十字绣单元
        # 每一路都采用两个attention模块，最后接一个线形层转化为最后的目标值
        self.attention1_1 = multi_heads_self_attention(feature_dim=length)
        self.attention1_2 = multi_heads_self_attention(feature_dim=length)
        self.linear1_1 = nn.Linear(length, length)
        self.linear1_2 = nn.Linear(length, 1)

        self.attention2_1 = multi_heads_self_attention(feature_dim=length)
        self.attention2_2 = multi_heads_self_attention(feature_dim=length)
        self.linear2_1 = nn.Linear(length, length)
        self.linear2_2 = nn.Linear(length, 1)

        self.attention3_1 = multi_heads_self_attention(feature_dim=length)
        self.attention3_2 = multi_heads_self_attention(feature_dim=length)
        self.linear3_1 = nn.Linear(length, length)
        self.linear3_2 = nn.Linear(length, 1)

        self.attention4_1 = multi_heads_self_attention(feature_dim=length)
        self.attention4_2 = multi_heads_self_attention(feature_dim=length)
        self.linear4_1 = nn.Linear(length, length)
        self.linear4_2 = nn.Linear(length, 1)

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
        # 1 2路，第二阶段
        out1_2, _ = self.attention1_2(out1_1, out1_1, out1_1)
        out2_2, _ = self.attention2_2(out2_1, out2_1, out2_1)
        # 1 2路，第三阶段
        out1_3 = nn.functional.relu(self.linear1_1(out1_2))
        out1_3 = nn.functional.relu(self.linear1_2(out1_3))
        out2_3 = nn.functional.relu(self.linear2_1(out2_2))
        out2_3 = nn.functional.relu(self.linear2_2(out2_3))

        # 3 4路，第一阶段
        out3_1, _ = self.attention3_1(input, input, input)
        out4_1, _ = self.attention4_1(input, input, input)
        # 3 4路，第二阶段
        out3_2, _ = self.attention3_2(out3_1, out3_1, out3_1)
        out4_2, _ = self.attention4_2(out4_1, out4_1, out4_1)
        # 3 4路，第三阶段
        out3_3 = nn.functional.relu(self.linear3_1(out3_2))
        out3_3 = nn.functional.relu(self.linear3_2(out3_3))
        out4_3 = nn.functional.relu(self.linear4_1(out4_2))
        out4_3 = nn.functional.relu(self.linear4_2(out4_3))

        # 分任务计算损失
        loss1 = self.loss1(torch.squeeze(out1_3), torch.squeeze(y1))
        loss2 = self.loss2(torch.squeeze(out2_3), torch.squeeze(y2))
        loss3 = self.loss3(torch.squeeze(out3_3), torch.squeeze(y3))
        loss4 = self.loss4(torch.squeeze(out4_3), torch.squeeze(y4))

        loss = self.lossLayer(loss1, loss2, loss3, loss4)

        return loss, out1_3, out2_3, out3_3, out4_3

# 3. 多任务去除加权损失函数结果
class model3(nn.Module):
    def __init__(self, length):
        super(model3, self).__init__()
        # 因为有四个任务，所以需要有四路模型，中间三个十字绣单元
        # 每一路都采用两个attention模块，最后接一个线形层转化为最后的目标值
        self.attention1_1 = multi_heads_self_attention(feature_dim=length)
        self.attention1_2 = multi_heads_self_attention(feature_dim=length)
        self.linear1_1 = nn.Linear(length, length)
        self.linear1_2 = nn.Linear(length, 1)

        self.attention2_1 = multi_heads_self_attention(feature_dim=length)
        self.attention2_2 = multi_heads_self_attention(feature_dim=length)
        self.linear2_1 = nn.Linear(length, length)
        self.linear2_2 = nn.Linear(length, 1)

        self.attention3_1 = multi_heads_self_attention(feature_dim=length)
        self.attention3_2 = multi_heads_self_attention(feature_dim=length)
        self.linear3_1 = nn.Linear(length, length)
        self.linear3_2 = nn.Linear(length, 1)

        self.attention4_1 = multi_heads_self_attention(feature_dim=length)
        self.attention4_2 = multi_heads_self_attention(feature_dim=length)
        self.linear4_1 = nn.Linear(length, length)
        self.linear4_2 = nn.Linear(length, 1)

        self.cross_stitch1 = cross_stitch(length*2)
        self.cross_stitch2 = cross_stitch(length*2)

        self.loss1 = nn.MSELoss()
        self.loss2 = nn.MSELoss()
        self.loss3 = nn.MSELoss()
        self.loss4 = nn.MSELoss()

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
        out1_3 = nn.functional.relu(self.linear1_1(out1_2))
        out1_3 = nn.functional.relu(self.linear1_2(out1_3))
        out2_3 = nn.functional.relu(self.linear2_1(out2_2))
        out2_3 = nn.functional.relu(self.linear2_2(out2_3))

        # 3 4路，第一阶段
        out3_1, _ = self.attention3_1(input, input, input)
        out4_1, _ = self.attention4_1(input, input, input)
        out3_1, out4_1 = self.cross_stitch2(out3_1, out4_1)
        # 3 4路，第二阶段
        out3_2, _ = self.attention3_2(out3_1, out3_1, out3_1)
        out4_2, _ = self.attention4_2(out4_1, out4_1, out4_1)
        # 3 4路，第三阶段
        out3_3 = nn.functional.relu(self.linear3_1(out3_2))
        out3_3 = nn.functional.relu(self.linear3_2(out3_3))
        out4_3 = nn.functional.relu(self.linear4_1(out4_2))
        out4_3 = nn.functional.relu(self.linear4_2(out4_3))

        # 分任务计算损失
        loss1 = self.loss1(torch.squeeze(out1_3), torch.squeeze(y1))
        loss2 = self.loss2(torch.squeeze(out2_3), torch.squeeze(y2))
        loss3 = self.loss3(torch.squeeze(out3_3), torch.squeeze(y3))
        loss4 = self.loss4(torch.squeeze(out4_3), torch.squeeze(y4))

        loss = loss1 + loss2 + loss3 + loss4

        return loss, out1_3, out2_3, out3_3, out4_3

# 4. 多任务去除attention结果
class model4(nn.Module):
    def __init__(self, length):
        super(model4, self).__init__()
        # 因为有四个任务，所以需要有四路模型，中间三个十字绣单元
        self.attention1_1 = nn.Linear(length, length)
        self.attention1_2 = nn.Linear(length, length)
        self.linear1_1 = nn.Linear(length, 1)
        # self.linear1_2 = nn.Linear(length, 1)

        self.attention2_1 = nn.Linear(length, length)
        self.attention2_2 = nn.Linear(length, length)
        self.linear2_1 = nn.Linear(length, 1)
        # self.linear2_2 = nn.Linear(length, 1)

        self.attention3_1 = nn.Linear(length, length)
        self.attention3_2 = nn.Linear(length, length)
        self.linear3_1 = nn.Linear(length, 1)
        # self.linear3_2 = nn.Linear(length, 1)

        self.attention4_1 = nn.Linear(length, length)
        self.attention4_2 = nn.Linear(length, length)
        self.linear4_1 = nn.Linear(length, 1)
        # self.linear4_2 = nn.Linear(length, 1)

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
        out1_1 = self.attention1_1(input)
        out2_1 = self.attention2_1(input)
        out1_1, out2_1 = self.cross_stitch1(out1_1, out2_1)
        # 1 2路，第二阶段
        out1_2 = self.attention1_2(out1_1)
        out2_2 = self.attention2_2(out2_1)
        # 1 2路，第三阶段
        out1_3 = nn.functional.relu(self.linear1_1(out1_2))
        # out1_3 = nn.functional.relu(self.linear1_2(out1_3))
        out2_3 = nn.functional.relu(self.linear2_1(out2_2))
        # out2_3 = nn.functional.relu(self.linear2_2(out2_3))

        # 3 4路，第一阶段
        out3_1 = self.attention3_1(input)
        out4_1 = self.attention4_1(input)
        out3_1, out4_1 = self.cross_stitch2(out3_1, out4_1)
        # 3 4路，第二阶段
        out3_2 = self.attention3_2(out3_1)
        out4_2 = self.attention4_2(out4_1)
        # 3 4路，第三阶段
        out3_3 = nn.functional.relu(self.linear3_1(out3_2))
        # out3_3 = nn.functional.relu(self.linear3_2(out3_3))
        out4_3 = nn.functional.relu(self.linear4_1(out4_2))
        # out4_3 = nn.functional.relu(self.linear4_2(out4_3))

        # 分任务计算损失
        loss1 = self.loss1(torch.squeeze(out1_3), torch.squeeze(y1))
        loss2 = self.loss2(torch.squeeze(out2_3), torch.squeeze(y2))
        loss3 = self.loss3(torch.squeeze(out3_3), torch.squeeze(y3))
        loss4 = self.loss4(torch.squeeze(out4_3), torch.squeeze(y4))

        loss = self.lossLayer(loss1, loss2, loss3, loss4)

        return loss, out1_3, out2_3, out3_3, out4_3

if __name__ =='__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    raw = read_dmax_data().values
    print(raw.shape)
    features = raw[:, :94]
    targets = raw[:, 94:]

    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.4)

    if torch.cuda.is_available():
        x_train = torch.from_numpy(x_train).to(device)
        x_test = torch.from_numpy(x_test).to(device)

        dmax_train = torch.from_numpy(y_train[:, 0]).view(-1, 1).to(device)
        dmax_test = torch.from_numpy(y_test[:, 0]).view(-1, 1).to(device)

        tg_train = torch.from_numpy(y_train[:, 1]).view(-1, 1).to(device)
        tg_test = torch.from_numpy(y_test[:, 1]).view(-1, 1).to(device)

        tx_train = torch.from_numpy(y_train[:, 2]).view(-1, 1).to(device)
        tx_test = torch.from_numpy(y_test[:, 2]).view(-1, 1).to(device)

        tl_train = torch.from_numpy(y_train[:, 3]).view(-1, 1).to(device)
        tl_test = torch.from_numpy(y_test[:, 3]).view(-1, 1).to(device)
    else:
        x_train = torch.from_numpy(x_train)
        x_test = torch.from_numpy(x_test)

        dmax_train = torch.from_numpy(y_train[:, 0]).view(-1, 1)
        dmax_test = torch.from_numpy(y_test[:, 0]).view(-1, 1)

        tg_train = torch.from_numpy(y_train[:, 1]).view(-1, 1)
        tg_test = torch.from_numpy(y_test[:, 1]).view(-1, 1)

        tx_train = torch.from_numpy(y_train[:, 2]).view(-1, 1)
        tx_test = torch.from_numpy(y_test[:, 2]).view(-1, 1)

        tl_train = torch.from_numpy(y_train[:, 3]).view(-1, 1)
        tl_test = torch.from_numpy(y_test[:, 3]).view(-1, 1)

    model = model2(length=x_train.size()[1])
    if torch.cuda.is_available():
        model.to(device)
    model.double()
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    criterion3 = nn.MSELoss()
    criterion4 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch_num = 50000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            loss, _, _, _, _ = model(x_train, dmax_train, tg_train, tx_train, tl_train)
            loss.backward()

            return loss

        optimizer.step(closure)

        if epoch % 10 == 9:
            print('epoch: {}'.format(epoch))

            loss, pred1, pred2, pred3, pred4 = model(x_test, dmax_test, tg_test, tx_test, tl_test)
            print('loss: {}'.format(loss))

            r21 = r2_score(torch.squeeze(pred1.cpu()).detach().numpy(), torch.squeeze(dmax_test.cpu()).detach().numpy())
            r22 = r2_score(torch.squeeze(pred2.cpu()).detach().numpy(), torch.squeeze(tg_test.cpu()).detach().numpy())
            r23 = r2_score(torch.squeeze(pred3.cpu()).detach().numpy(), torch.squeeze(tx_test.cpu()).detach().numpy())
            r24 = r2_score(torch.squeeze(pred4.cpu()).detach().numpy(), torch.squeeze(tl_test.cpu()).detach().numpy())
            r2 = r21 + r22 + r23 + r24
            print('r2_1: {}\nr2_2: {}\nr2_3: {}\nr2_4: {}\nr2: {}'.format(r21, r22, r23, r24, r2))
            if r2 >= 3.3:
                break