import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import math
import imblearn
from imblearn.over_sampling import SMOTE
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

class gfa_classifier(nn.Module):
    def __init__(self, feature_dim=56):
        super(gfa_classifier, self).__init__()

        self.layer_attention_1 = multi_heads_self_attention()
        self.layer_attention_2 = multi_heads_self_attention()

        self.linear_1 = nn.Linear(feature_dim, 512)
        self.linear_2 = nn.Linear(512, 256)
        self.linear_3 = nn.Linear(256, feature_dim)
        # self.linear_4 = nn.Linear(feature_dim, feature_dim)

        self.layer_final = nn.Linear(feature_dim, 3)

    def forward(self, context):
        # pass through attention
        output, attention = self.layer_attention_1(context, context, context)
        output, _ = self.layer_attention_2(output, output, output)
        # output = self.layer_norm_2(residual + output)

        # pass through linear
        output = nn.functional.relu(self.linear_1(output))
        output = nn.functional.relu(self.linear_2(output))
        output = nn.functional.relu(self.linear_3(output))

        # output = nn.functional.relu(self.linear_4(output))

        # pass through layer final
        no_softmax = self.layer_final(output)
        # softmax = nn.functional.softmax(self.layer_final(output), dim=-1)

        return no_softmax

class mlp_classifier(nn.Module):
    def __init__(self, feature_dim=56):
        super(mlp_classifier, self).__init__()

        self.linear_1 = nn.Linear(feature_dim, 512)
        self.layer_norm_1 = nn.LayerNorm(512)
        self.linear_2 = nn.Linear(512, 256)
        self.layer_norm_2 = nn.LayerNorm(256)
        self.linear_3 = nn.Linear(256, 3)

    def forward(self, context):
        output = nn.functional.relu(self.linear_1(context))
        output = self.layer_norm_1(output)
        output = nn.functional.relu(self.linear_2(output))
        output = self.layer_norm_2(output)
        output = nn.functional.relu(self.linear_3(output))

        return output

if __name__ == '__main__':
    with open('/home/lab106/zy/MatTime/GFA_trans.pk', 'rb') as f:
        raw = pickle.load(f)
    features = raw.iloc[:, :-4].values
    target = raw.iloc[:, -1:].values

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.5)
    # smote_trans = SMOTE(sampling_strategy='auto', random_state=1219)
    # x_train, y_train = smote_trans.fit_resample(x_train, y_train)
    batch_size = 1
    features_size = 56
    # target_size = 3

    device = torch.device('cuda:0')
    x_train = torch.from_numpy(x_train).view(batch_size, -1, features_size).to(device)
    x_test = torch.from_numpy(x_test).view(batch_size, -1, features_size).to(device)
    y_train = torch.from_numpy(y_train).view(batch_size, -1).long().to(device)
    y_test = torch.from_numpy(y_test).view(batch_size, -1).long().to(device)

    # # 多头注意力
    # multi_att = multi_heads_self_attention()
    # multi_att.to(device)
    # multi_att.double()
    # # criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(multi_att.parameters())

    # epoch_num = 400

    # for epoch in range(epoch_num):
    #     def closure():
    #         optimizer.zero_grad()
    #         out, _ = multi_att(x_train, x_train, x_train)
    #         loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
    #         # print('loss:', loss.data.item())
    #         # loss_list.append(loss.data.item())
    #         loss.backward()
    #         return loss

    #     optimizer.step(closure)

    #     if epoch % 10 == 9:
    #         print('epoch : ', epoch)
    #         pred, _ = multi_att(x_test, x_test, x_test)
    #         loss = criterion(torch.squeeze(pred), torch.squeeze(y_test))
    #         print('test loss:', loss.data.item())
    #         pred = torch.topk(pred.data.cpu(), 1)[1].squeeze()
    #         target_names = ['BMG', 'CRA', 'RMG']
    #         # print('r2:', r2_score(torch.squeeze(y_test.cpu()), torch.squeeze(pred.data.cpu())))
    #         print(classification_report(torch.squeeze(y_test.cpu()), pred, target_names=target_names))

    # gfa = gfa_classifier()
    # gfa.to(device)
    # gfa.double()
    mlp = mlp_classifier()
    mlp.to(device)
    mlp.double()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.00001)

    epoch_num = 100000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            out = mlp(x_train)
            loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
            # print('loss:', loss.data.item())
            # loss_list.append(loss.data.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        if epoch % 100 == 99:
            print('epoch : ', epoch)
            pred = mlp(x_test)
            loss = criterion(torch.squeeze(pred), torch.squeeze(y_test))
            print('test loss:', loss.data.item())
            pred = torch.topk(pred.data.cpu(), 1)[1].squeeze()

            # 参考 focal loss
            focus_param = 1
            target_names = ['BMG', 'CRA', 'RMG']
            pro_list = f1_score(torch.squeeze(y_test.cpu()), pred, average=None)
            BMG_weight = (2 - pro_list[0]) ** focus_param
            CRA_weight = 7 * (2 - pro_list[1]) ** focus_param
            RMG_weight = 6 * (2 - pro_list[2]) ** focus_param
            # BMG_weight = math.e ** (1 - pro_list[0])
            # CRA_weight = math.e ** (1 - pro_list[0]) + 1
            # RMG_weight = math.e ** (1 - pro_list[0])
            # BMG_weight = 1
            # RMG_weight = 1
            weight = torch.tensor([BMG_weight, CRA_weight, RMG_weight], dtype=torch.float64).to(device)
            criterion.weight = weight
            print(classification_report(torch.squeeze(y_test.cpu()), pred, target_names=target_names))