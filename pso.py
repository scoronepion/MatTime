import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sko.PSO import PSO

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

def calc_func(x):
    input = np.zeros(45)
    # 修改元素
    input[2] = x[0]
    input[21] = x[1]
    input[22] = x[2]
    input[23] = x[3]
    input[32] = x[4]
    input[34] = x[5]
    input[41] = x[6]
    input[44] = x[7]
    x = input
    # 归一化
    x = x / x.sum()
    x = torch.from_numpy(x).view(1, -1)
    model = torch.load('models/embedding_attention_08502.bin', map_location='cpu')
    return -torch.squeeze(model(x)).detach().numpy()

def result_process():
    #     best_x is  [0.13634683 0.15337651 1.         1.         1.         1.
    #  0.         1.         0.07944989 0.         0.         0.
    #  0.         0.         0.03435112 0.29926256 0.         0.08802135
    #  0.09057049 0.         0.         1.         1.         0.
    #  1.         0.12208753 0.         0.         0.         0.
    #  0.         0.         0.         1.         1.         0.12815723
    #  0.51435195 1.         0.         0.01084121 0.         0.00844453
    #  1.         0.         1.        ] best_y is -32.22407266552083

    #     best_x is  [ 0.          0.         10.         10.         10.          6.16798118
    #   0.         10.          0.56373507  1.63345313  0.          0.
    #   0.          0.          0.          0.          8.74504363  1.59825373
    #   0.          0.          0.         10.          0.          0.
    #   0.85799561  2.89566099 10.          0.          0.          0.
    #  10.          6.02697363  0.          0.          7.30377607  0.
    #   3.67391764 10.          1.43108727 10.          0.          0.
    #  10.          0.          7.60143824] best_y is -31.632656606539236

    #     best_x is  [ 56.74030825   0.           0.         100.         100.
    #  100.           0.         100.          29.05084393   0.
    #  100.           0.           0.         100.           0.
    #    0.         100.          30.33113793   0.           0.
    #    0.         100.         100.           0.         100.
    #  100.          67.17916563   0.          46.41100105   0.
    #  100.          91.22520903 100.          78.5072529  100.
    #   55.09179187 100.           0.          26.76232401   0.
    #    0.           0.          99.34618466   0.         100.        ] best_y is -31.047092573646005

    data = np.array([56.74030825,0.,0.,100.,100.,100.,0.,100.,29.05084393,0.,100.,0.,0.,100.,0.,0.,100.,30.33113793,0.,0.,0.,100.,100.,0.,100.,
                    100.,67.17916563,0.,46.41100105,0.,100.,91.22520903,100.,78.5072529,100.,55.09179187,100.,0.,26.76232401,0.,0.,0.,99.34618466,0.,100.])
    data /= data.sum()
    print(data)

if __name__ == '__main__':
    # Top element (start from 0): Cu-2, Al-22, Zr-23, Ni-21, Fe-7, Mg-39, B-19
    # Tail 100 top element (start from 0): Cu-2, La-21, Al-22, Zr-23, Mg-32, Y-34, Ag-41, Co-44
    # pso = PSO(func=calc_func, dim=8, pop=400, max_iter=200, lb=np.zeros(8), ub=np.ones(8)*100)
    # pso.run()
    # print('best_x is ', pso.gbest_x, 'best_y is', pso.gbest_y)
    # result_process()

    raw = np.array([0., 11.64490368, 0., 100., 8.24245089, 25.9136744, 8.70419478, 21.68054529])
    print(raw / raw.sum())

    # tail result 
    # 1. raw = [0., 11.64490368, 0., 100., 8.24245089, 25.9136744, 8.70419478, 21.68054529]
    #    Cu:0, La:0.0661, Al:0, Zr:0.5675, Mg:0.0468, Y:0.1471, Ag:0.0494, Co:0.1231
    #    La