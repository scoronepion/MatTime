import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# 1. drop sTemper and sHour，102条；测试集0.3；deltaT 0.96 deltaL 0.98 alpha 0.87
# 2. dropna；81条；测试集0.3；不收敛
# 3. deltaT 0.17 alpha 0.96 T1 0.95
# 4. drop deltaT
# 5. alpha 0.90  t1 0.92  t2 0.94

minv = []
maxv = []

# samples
# samples = [
#     [3.1, 0, 0, 0, 0, 0, 0, 0, 0.5, 0.4, 0, 0, 0, 179.45, -330.00, 2300.00, 1970.00, -6.73469387755102, 390.15, 439.15],
#     [3.1, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.5, 0, 0, 0, 417.49, -2982.00, 1042.00, -1940, -53.25, 314.15, 370.15],
#     [3.1, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.6, 0, 0, 0, 400.85, -1420.00, 1100.00, -320.00, -27.84313725490196, 330.15, 381.15],
#     [3.2, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.4, 0, 0, 0, 183.15, -500.00, 1800.00, 1300.00, -8.928571428571429, 358.15, 414.15],
#     [3.2, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.5, 0, 0, 0, 291.80, -5534.40, 1264.4, -4270, -106.43076923076923, 292.15, 344.15],
#     [3.2, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.6, 0, 0, 0, 381.25, -2114.00, 8140, -1300.00, -30.637681159420303, 239.15, 308.15],
#     [3.3, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.4, 0, 0, 0, 347.22, -138, 63, -75, -3.1363636363636362, 262.15, 306.15],
#     [3.3, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.5, 0, 0, 0, 346.52, -152, 6.5, -145.5, -2.3109786690587324, 206.15, 271.92],
#     [3.3, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.6, 0, 0, 0, 263.15, -121.7, 59.2, -62.5, -2.830232558139535, 175.15, 218.15]
# ]

samples = [
    [3.3, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.4, 0, 0, 0, 347.22, -138, 63, -75, -3.1363636363636362, 262.15, 306.15]
]




def trans_sample():
    global samples
    # for item in samples:
    #     print(item[-6] / (item[-1] - item[-2]))

    for i in range(len(samples)):
        for j in range(7):
            samples[i][j + 13] = (samples[i][j + 13] - minv[j]) / (maxv[j] - minv[j])


def read_data():
    raw = pd.read_csv('/home/lab106/zy/new_geogre_data/sup_exp/wyjData.csv')
    raw.drop(columns=['id', 'pic_num', 'formula', 'sTemper', 'sHour', 'deltaT'], inplace=True)
    # 归一化
    max_min_scaler = lambda x : (x-np.min(x))/(np.max(x)-np.min(x))

    raw = raw.dropna()

    # raw['sTemper'] = raw[['sTemper']].apply(max_min_scaler)
    # raw['sHour'] = raw[['sHour']].apply(max_min_scaler)

    global minv, maxv
    orders = ['t0', 'deltaL', 'l1', 'l2', 'alpha', 't1', 't2']
    for i in range(len(orders)):
        minv.append(raw[orders[i]].min())
        maxv.append(raw[orders[i]].max())

    raw['t0'] = raw[['t0']].apply(max_min_scaler)
    raw['t1'] = raw[['t1']].apply(max_min_scaler)
    raw['t2'] = raw[['t2']].apply(max_min_scaler)
    raw['l1'] = raw[['l1']].apply(max_min_scaler)
    raw['l2'] = raw[['l2']].apply(max_min_scaler)
    # raw['deltaT'] = raw[['deltaT']].apply(max_min_scaler)
    raw['deltaL'] = raw[['deltaL']].apply(max_min_scaler)
    raw['alpha'] = raw[['alpha']].apply(max_min_scaler)
    return raw.dropna().astype('float64')

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
        
    def forward(self, loss0, loss1, loss2):
        # loss0
        factor0 = torch.div(1.0, torch.mul(self._sigmas_sq[0], 2.0))
        loss = torch.add(torch.mul(factor0, loss0), 0.5 * torch.log(self._sigmas_sq[0]))
        # loss1
        factor1 = torch.div(1.0, torch.mul(self._sigmas_sq[1], 2.0))
        loss = torch.add(loss, torch.add(torch.mul(factor1, loss1), 0.5 * torch.log(self._sigmas_sq[1])))
        # loss2
        factor2 = torch.div(1.0, torch.mul(self._sigmas_sq[2], 2.0))
        loss = torch.add(loss, torch.add(torch.mul(factor2, loss2), 0.5 * torch.log(self._sigmas_sq[2])))

        return loss

class MultiLossLayer_other(nn.Module):
    def __init__(self, list_length):
        super(MultiLossLayer_other, self).__init__()
        self._sigmas_sq = nn.ParameterList([nn.Parameter(torch.empty(())) for i in range(list_length)])
        for p in self.parameters():
            nn.init.uniform_(p, 0.5, 0.8)
        
    def forward(self, loss0, loss1):
        # loss0
        factor0 = torch.div(1.0, torch.mul(self._sigmas_sq[0], 2.0))
        loss = torch.add(torch.mul(factor0, loss0), 0.5 * torch.log(self._sigmas_sq[0]))
        # loss1
        factor1 = torch.div(1.0, torch.mul(self._sigmas_sq[1], 2.0))
        loss = torch.add(loss, torch.add(torch.mul(factor1, loss1), 0.5 * torch.log(self._sigmas_sq[1])))

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
        input1_reshaped = torch.squeeze(input1, dim=0)
        input2_reshaped = torch.squeeze(input2, dim=0)
        input_reshaped = torch.cat([input1_reshaped, input2_reshaped], dim=-1)
        output = torch.matmul(input_reshaped, self.matrix)

        output1 = torch.reshape(output[:, :input1.size()[-1]], input1.size())
        output2 = torch.reshape(output[:, input2.size()[-1]:], input2.size())

        return output1, output2

class CrossModel_other(nn.Module):
    def __init__(self, length):
        super(CrossModel_other, self).__init__()
        # 因为有三个任务，所以需要有三路模型，中间两个十字绣单元
        # 每一路都采用两个attention模块，最后接一个线形层转化为最后的目标值
        self.attention1_1 = multi_heads_self_attention(feature_dim=length)
        self.attention1_2 = multi_heads_self_attention(feature_dim=length)
        self.linear1 = nn.Linear(length, 1)

        self.attention2_1 = multi_heads_self_attention(feature_dim=length)
        self.attention2_2 = multi_heads_self_attention(feature_dim=length)
        self.linear2 = nn.Linear(length, 1)

        self.cross_stitch1 = cross_stitch(length*2)

        self.loss1 = nn.MSELoss()
        self.loss2 = nn.MSELoss()

        self.lossLayer = MultiLossLayer_other(list_length=2)

    def forward(self, input, y1, y2):
        # y1=fatigue; y2=tensile; y3=fracture; y4=hardness;
        # 统一描述：out_路编号_阶段编号

        # 第一阶段
        out1_1, _ = self.attention1_1(input, input, input)
        out2_1, _ = self.attention2_1(input, input, input)
        out1_1, out2_1 = self.cross_stitch1(out1_1, out2_1)

        # 第二阶段
        out1_2, _ = self.attention1_2(out1_1, out1_1, out1_1)
        out2_2, _ = self.attention2_2(out2_1, out2_1, out2_1)

        # 第三阶段
        out1_3 = nn.functional.relu(self.linear1(out1_2))
        out2_3 = nn.functional.relu(self.linear2(out2_2))

        # 分任务计算损失
        loss1 = self.loss1(torch.squeeze(out1_3), torch.squeeze(y1))
        loss2 = self.loss2(torch.squeeze(out2_3), torch.squeeze(y2))

        loss = self.lossLayer(loss1, loss2)

        return loss, out1_3, out2_3

class CrossModel(nn.Module):
    def __init__(self, length):
        super(CrossModel, self).__init__()
        # 因为有三个任务，所以需要有三路模型，中间两个十字绣单元
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

        self.cross_stitch1 = cross_stitch(length*2)
        self.cross_stitch2 = cross_stitch(length*2)

        self.loss1 = nn.MSELoss()
        self.loss2 = nn.MSELoss()
        self.loss3 = nn.MSELoss()

        self.lossLayer = MultiLossLayer(list_length=3)

    def forward(self, input, y1, y2, y3):
        # y1=fatigue; y2=tensile; y3=fracture; y4=hardness;
        # 统一描述：out_路编号_阶段编号

        # 第一阶段
        out1_1, _ = self.attention1_1(input, input, input)
        out2_1, _ = self.attention2_1(input, input, input)
        out3_1, _ = self.attention3_1(input, input, input)
        out1_1, out2_1_a = self.cross_stitch1(out1_1, out2_1)
        out2_1_b, out3_1 = self.cross_stitch2(out2_1, out3_1)
        out2_1 = 0.5 * out2_1_a + 0.5 * out2_1_b
        # 第二阶段
        out1_2, _ = self.attention1_2(out1_1, out1_1, out1_1)
        out2_2, _ = self.attention2_2(out2_1, out2_1, out2_1)
        out3_2, _ = self.attention3_2(out3_1, out3_1, out3_1)
        # 第三阶段
        out1_3 = nn.functional.relu(self.linear1(out1_2))
        out2_3 = nn.functional.relu(self.linear2(out2_2))
        out3_3 = nn.functional.relu(self.linear3(out3_2))

        # 分任务计算损失
        loss1 = self.loss1(torch.squeeze(out1_3), torch.squeeze(y1))
        loss2 = self.loss2(torch.squeeze(out2_3), torch.squeeze(y2))
        loss3 = self.loss3(torch.squeeze(out3_3), torch.squeeze(y3))

        loss = self.lossLayer(loss1, loss2, loss3)

        return loss, out1_3, out2_3, out3_3

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    raw = read_data().values
    print(raw.shape)
    features = raw[:, :19]
    targets = raw[:, 19:]

    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.3)
    # x_train, x_test = features, features
    # y_train, y_test = targets, targets

    if torch.cuda.is_available():
        x_train = torch.from_numpy(x_train).to(device)
        x_test = torch.from_numpy(x_test).to(device)

        alpha_train = torch.from_numpy(y_train[:, 0]).view(-1, 1).to(device)
        alpha_test = torch.from_numpy(y_test[:, 0]).view(-1, 1).to(device)

        t1_train = torch.from_numpy(y_train[:, 0]).view(-1, 1).to(device)
        t1_test = torch.from_numpy(y_test[:, 0]).view(-1, 1).to(device)

        t2_train = torch.from_numpy(y_train[:, 1]).view(-1, 1).to(device)
        t2_test = torch.from_numpy(y_test[:, 1]).view(-1, 1).to(device)
    else:
        x_train = torch.from_numpy(x_train)
        x_test = torch.from_numpy(x_test)

        alpha_train = torch.from_numpy(y_train[:, 0]).view(-1, 1)
        alpha_test = torch.from_numpy(y_test[:, 0]).view(-1, 1)

        t1_train = torch.from_numpy(y_train[:, 0]).view(-1, 1)
        t1_test = torch.from_numpy(y_test[:, 0]).view(-1, 1)

        t2_train = torch.from_numpy(y_train[:, 1]).view(-1, 1)
        t2_test = torch.from_numpy(y_test[:, 1]).view(-1, 1)

    model = CrossModel(length=x_train.size()[1])
    if torch.cuda.is_available():
        model.to(device)
    model.double()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch_num = 50000

    loger1 = open('/home/lab106/zy/new_geogre_data/sup_exp/logs/1.txt', 'w')
    loger2 = open('/home/lab106/zy/new_geogre_data/sup_exp/logs/2.txt', 'w')
    loger3 = open('/home/lab106/zy/new_geogre_data/sup_exp/logs/3.txt', 'w')

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            loss, _, _, _ = model(x_train, alpha_train, t1_train, t2_train)
            loss.backward()

            return loss

        optimizer.step(closure)

        if epoch % 10 == 9:
            print('epoch: {}'.format(epoch))

            loss, pred1, pred2, pred3 = model(x_test, alpha_test, t1_test, t2_test)
            print('loss: {}'.format(loss))

            r21 = r2_score(torch.squeeze(pred1.cpu()).detach().numpy(), torch.squeeze(alpha_test.cpu()).detach().numpy())
            r22 = r2_score(torch.squeeze(pred2.cpu()).detach().numpy(), torch.squeeze(t1_test.cpu()).detach().numpy())
            r23 = r2_score(torch.squeeze(pred3.cpu()).detach().numpy(), torch.squeeze(t2_test.cpu()).detach().numpy())
            r2 = r21 + r22 + r23
            print('r2_1: {}\nr2_2: {}\nr2_3: {}\nr2: {}'.format(r21, r22, r23, r2))
            loger1.writelines(str(r21) + '\n')
            loger2.writelines(str(r22) + '\n')
            loger3.writelines(str(r23) + '\n')
            # if r2 >= 2.85:
            #     trans_sample()

            #     for item in samples:
            #         samFea = np.array(item[:-3])
            #         samFea = torch.from_numpy(samFea).view(1, -1).to(device)
                    
            #         samTarAlpha = np.array(item[-3])
            #         samTarT1 = np.array(item[-2])
            #         samTarT2 = np.array(item[-1])

            #         samTarAlpha = torch.from_numpy(samTarAlpha).view(1, -1).to(device)
            #         samTarT1 = torch.from_numpy(samTarT1).view(1, -1).to(device)
            #         samTarT2 = torch.from_numpy(samTarT2).view(1, -1).to(device)

            #         _, alphaPred, t1Pred, t2Pred = model(samFea, samTarAlpha, samTarT1, samTarT2)

            #         alphaPred = torch.squeeze(alphaPred.cpu()).detach().numpy()
            #         t1Pred = torch.squeeze(t1Pred.cpu()).detach().numpy()
            #         t2Pred = torch.squeeze(t2Pred.cpu()).detach().numpy()

            #         # alphaPred = alphaPred * (maxv[-3] - minv[-3]) + minv[-3]
            #         # t1Pred = t1Pred * (maxv[-2] - minv[-2]) + minv[-2]
            #         # t2Pred = t2Pred * (maxv[-1] - minv[-1]) + minv[-1]

            #         print('pred alpha = {}, real alpha = {}'.format(alphaPred, item[-3]))
            #         print('pred t1 = {}, real t1 = {}'.format(t1Pred, item[-2]))
            #         print('pred t2 = {}, real t2 = {}'.format(t2Pred, item[-1]))

            #     break