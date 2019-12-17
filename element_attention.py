import torch
import torch.nn as nn
import torch.optim as optim
from predata import read_element
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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
class multi_heads_self_attention_rnn(nn.Module):
    def __init__(self, feature_dim=45, num_heads=3, dropout=0.0):
        super(multi_heads_self_attention_rnn, self).__init__()

        self.dim_per_head = feature_dim // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(feature_dim, self.dim_per_head * num_heads)
        self.linear_k = nn.Linear(feature_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(feature_dim, self.dim_per_head * num_heads)

        self.compress_size = 5
        self.rnn_hidden_size = 264
        self.rnn_input_size = self.compress_size + 3

        self.sdp_attention = scaled_dot_product_attention(dropout)
        self.linear_attention = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(feature_dim)

        self.layer_compress = nn.Linear(feature_dim, self.compress_size)
        self.layer_rnn = nn.GRU(input_size=self.rnn_input_size, hidden_size=self.rnn_hidden_size, num_layers=1)
        # self.linear_lstm = nn.Linear(264, 64)
        self.linear_final = nn.Linear(self.rnn_hidden_size, 1)

    def forward(self, key, value, query, temperature):
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

        # compress element dimensions
        output = self.layer_compress(output)

        # concat element vector and temperature vector
        output = torch.cat((output, temperature), -1)

        # pass through LSTM
        output, _ = self.layer_rnn(output)

        # pass through linear
        # output = nn.functional.relu(self.linear_lstm(output))
        output = self.linear_final(output)

        return output, attention

if __name__ == '__main__':
    raw, _ = read_element(noise=False, sort=True, rare_element_scaler=2)
    features = raw.iloc[:, :-1].values
    target = raw.iloc[:, -1:].values

    batch_size = 1
    element_size = 45
    temperature_size = 3

    # x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3)
    # print(x_train.shape)
    # print(x_test.shape)

    train_size = int(raw.shape[0] * 0.7)

    train_element = features[:train_size, :-3]
    train_temperature = features[:train_size, -3:]
    train_target = target[:train_size, :]

    test_element = features[train_size:, :-3]
    test_temperature = features[train_size:, -3:]
    test_target = target[train_size:, :]

    device = torch.device('cuda:0')
    # train set
    x_element_train = torch.from_numpy(train_element).view(batch_size, -1, element_size).to(device)
    x_temperature_train = torch.from_numpy(train_temperature).view(batch_size, -1, temperature_size).to(device)
    y_train = torch.from_numpy(train_target).view(batch_size, -1, 1).to(device)
    # test set
    x_element_test = torch.from_numpy(test_element).view(batch_size, -1, element_size).to(device)
    x_temperature_test = torch.from_numpy(test_temperature).view(batch_size, -1, temperature_size).to(device)
    y_test = torch.from_numpy(test_target).view(batch_size, -1, 1).to(device)

    multi_att_rnn = multi_heads_self_attention_rnn()
    multi_att_rnn.to(device)
    multi_att_rnn.double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(multi_att_rnn.parameters())

    epoch_num = 2000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            out, _ = multi_att_rnn(x_element_train, x_element_train, x_element_train, x_temperature_train)
            loss = criterion(out, y_train)
            # print('loss:', loss.data.item())
            # loss_list.append(loss.data.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        if epoch % 10 == 9:
            print('epoch : ', epoch)
            pred, _ = multi_att_rnn(x_element_test, x_element_test, x_element_test, x_temperature_test)
            loss = criterion(pred, y_test)
            print('test loss:', loss.data.item())
            print('r2:', r2_score(torch.squeeze(y_test.cpu()), torch.squeeze(pred.data.cpu())))