import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
# from sko.PSO import PSO
from big_predata import read_element, read_over_element, read_cmp
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class chemical_embedding(nn.Module):
    '''返回元素 embedding 表示'''
    def __init__(self, length, embedding_size):
        '''length: 元素数量；embedding_size: 嵌入大小'''
        super(chemical_embedding, self).__init__()
        self.length = length
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(length, embedding_size)

    def forward(self, input):
        # if torch.cuda.is_available():
        #     device = torch.device('cuda:0')

        index = np.tile([i for i in range(self.length)], (input.size(0), 1))

        # if torch.cuda.is_available():
        #     index = torch.tensor(index, dtype=torch.long).to(device)
        # else:
        #     index = torch.tensor(index, dtype=torch.long)
        index = torch.tensor(index, dtype=torch.long)

        embed = self.embedding(index).view(-1)
        # 输入变换
        # if torch.cuda.is_available():
        #     trans = torch.zeros(self.length, self.length * self.embedding_size, dtype=torch.float64).to(device)
        # else:
        #     trans = torch.zeros(self.length, self.length * self.embedding_size, dtype=torch.float64)
        trans = torch.zeros(self.length, self.length * self.embedding_size, dtype=torch.float64)

        flag = 0
        for i in range(self.length):
            for j in range(self.embedding_size):
                trans[i][flag + j] = 1
            flag += self.embedding_size
        expanded_data = torch.einsum('ij,jk->ik', [input, trans]).view(-1)
        result = (expanded_data * embed).view(input.size(0), 1, -1)

        return result

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
    def __init__(self, length, embedding_size):
        super(embedding_attention, self).__init__()
        self.embedding = chemical_embedding(length=length, embedding_size=embedding_size)
        self.attention = multi_heads_self_attention(feature_dim=length * embedding_size, num_heads=3)
        self.linear_final = nn.Linear(length * embedding_size, 1)

    def forward(self, input):
        embed = self.embedding(input)
        output, _ = self.attention(embed, embed, embed)
        output = self.linear_final(output)

        return output

def calc_r2():
    raw = read_cmp().values
    features = raw[:-1, :-1]
    target = raw[:-1, -1:]
    model = torch.load('models/embedding_attention_Full_Dmax_no75_scaled_07952.bin', map_location='cpu')
    df = pd.DataFrame()
    train_r2_list = []
    test_r2_list = []
    epoch = 500
    while epoch > 0:
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1)
        x_train = torch.from_numpy(x_train)
        x_test = torch.from_numpy(x_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
        train_pred = torch.squeeze(model(x_train)).detach().numpy()
        test_pred = torch.squeeze(model(x_test)).detach().numpy()

        train_r2 = r2_score(torch.squeeze(y_train).detach().numpy(), train_pred)
        test_r2 = r2_score(torch.squeeze(y_test).detach().numpy(), test_pred)
        # print(train_r2)
        # print(test_r2)
        train_r2_list.append(train_r2)
        test_r2_list.append(test_r2)
        epoch -= 1

        # train_df = pd.DataFrame()
        # train_df['y_train'] = torch.squeeze(y_train).detach().numpy()
        # train_df['train_pred'] = train_pred
        # test_df = pd.DataFrame()
        # test_df['y_test'] = torch.squeeze(y_test).detach().numpy()
        # test_df['test_pred'] = test_pred
        # train_df.to_csv('Full_Dmax_train_pred.csv', index=False)
        # test_df.to_csv('Full_Dmax_test_pred.csv', index=False)

    df['train_r2'] = train_r2_list
    df['test_r2'] = test_r2_list
    df.to_csv('scaled_500_r2.csv', index=False)

if __name__ == '__main__':
    # train_df = pd.read_csv('Full_Dmax_test_pred.csv')
    # print(train_df.info())
    # plt = sns.lmplot(x="y_test", y="test_pred", data=train_df)
    # plt.savefig('2.png')
    calc_r2()