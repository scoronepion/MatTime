import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from big_predata import read_element, read_pro_features
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

class cross_layer(nn.Module):
    def __init__(self, input_dim):
        super(cross_layer, self).__init__()
        self.w = nn.Parameter(torch.empty((input_dim)))
        self.b = nn.Parameter(torch.empty((input_dim)))

        for p in self.parameters():
            nn.init.uniform_(p, 0.2, 1)

    def forward(self, x0, x):
        trans = torch.einsum('bi,i->b', [x, self.w])
        x_trans = torch.einsum('bi,b->bi', [x0, trans])

        return x_trans + self.b + x

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
        output = torch.squeeze(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        # # pass through linear
        # output = nn.functional.relu(self.linear_1(output))
        # output = nn.functional.relu(self.linear_2(output))

        # # pass through layer final
        # output = self.layer_final(output)

        return output, attention

class mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(mlp, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, output_dim)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        output = nn.functional.relu(self.linear1(input))
        # output = self.dropout(output)
        output = nn.functional.relu(self.linear2(output))
        # output = self.dropout(output)
        output = nn.functional.relu(self.linear3(output))
        # output = self.dropout(output)
        output = self.linear4(output)

        return output

class cross_mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cross_mlp, self).__init__()
        self.cross1 = cross_layer(input_dim)
        self.cross2 = cross_layer(input_dim)
        self.cross3 = cross_layer(input_dim)
        self.cross4 = cross_layer(input_dim)
        self.mlp = mlp(input_dim, output_dim)
        # self.final_linear = nn.Linear(input_dim + output_dim, 1)
        self.linear1 = nn.Linear(input_dim + output_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        cross_out = self.cross1(input, input)
        cross_out = self.cross2(input, cross_out)
        cross_out = self.cross3(input, cross_out)
        cross_out = self.cross4(input, cross_out)
        mlp_out = self.mlp(input)
        cat_res = torch.cat((cross_out, mlp_out), dim=-1)
        # output = self.final_linear(cat_res)
        output = nn.functional.relu(self.linear1(cat_res))
        # output = self.dropout(output)
        # output = nn.functional.relu(self.linear2(output))
        output = self.dropout(output)
        output = self.linear3(output)

        return output

class cross_attention(nn.Module):
    def __init__(self, input_dim):
        super(cross_attention, self).__init__()
        self.cross1 = cross_layer(input_dim)
        self.cross2 = cross_layer(input_dim)
        self.cross3 = cross_layer(input_dim)
        self.cross4 = cross_layer(input_dim)
        self.attention_layer = multi_heads_self_attention(feature_dim=input_dim, num_heads=2)
        self.linear1 = nn.Linear(input_dim * 2, 256)
        self.final_linear = nn.Linear(256, 1)

    def forward(self, input):
        cross_out = self.cross1(input, input)
        cross_out = self.cross2(input, cross_out)
        cross_out = self.cross3(input, cross_out)
        cross_out = self.cross4(input, cross_out)
        attention_out, _ = self.attention_layer(input, input, input)
        cat_res = torch.cat((cross_out, attention_out), dim=-1)
        output = nn.functional.relu(self.linear1(cat_res))
        output = self.final_linear(output)

        return output

class cross_attention_attention(nn.Module):
    def __init__(self, input_dim):
        super(cross_attention_attention, self).__init__()
        self.cross1 = cross_layer(input_dim)
        self.cross2 = cross_layer(input_dim)
        self.cross3 = cross_layer(input_dim)
        self.cross4 = cross_layer(input_dim)
        self.attention1 = multi_heads_self_attention(feature_dim=input_dim, num_heads=2)
        self.attention2 = multi_heads_self_attention(feature_dim=input_dim*2, num_heads=2)
        self.linear1 = nn.Linear(input_dim * 2, 256)
        self.final_linear = nn.Linear(256, 1)

    def forward(self, input):
        cross_out = self.cross1(input, input)
        cross_out = self.cross2(input, cross_out)
        cross_out = self.cross3(input, cross_out)
        cross_out = self.cross4(input, cross_out)
        attention_out, _ = self.attention1(input, input, input)
        cat_res = torch.cat((cross_out, attention_out), dim=-1)
        output, _ = self.attention2(cat_res, cat_res, cat_res)
        output = nn.functional.relu(self.linear1(output))
        output = self.final_linear(output)

        return output

class pure_cross(nn.Module):
    def __init__(self, input_dim):
        super(pure_cross, self).__init__()
        self.cross1 = cross_layer(input_dim)
        self.cross2 = cross_layer(input_dim)
        self.cross3 = cross_layer(input_dim)
        self.cross4 = cross_layer(input_dim)
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 1)

    def forward(self, input):
        cross_out = self.cross1(input, input)
        cross_out = self.cross2(input, cross_out)
        cross_out = self.cross3(input, cross_out)
        cross_out = self.cross4(input, cross_out)
        output = nn.functional.relu(self.linear1(cross_out))
        output = self.linear2(output)

        return output

class pure_attention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(pure_attention, self).__init__()
        self.attention = multi_heads_self_attention(feature_dim=input_dim, num_heads=num_heads)
        self.linear = nn.Linear(input_dim, 256)
        self.final_linear = nn.Linear(256, 1)

    def forward(self, input):
        output, _ = self.attention(input, input, input)
        output = nn.functional.relu(self.linear(output))
        output = self.final_linear(output)

        return output

class cross_mlp_attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cross_mlp_attention, self).__init__()
        self.cross1 = cross_layer(input_dim)
        self.cross2 = cross_layer(input_dim)
        self.cross3 = cross_layer(input_dim)
        self.cross4 = cross_layer(input_dim)
        self.mlp = mlp(input_dim, output_dim)
        self.attention = multi_heads_self_attention(feature_dim=input_dim + output_dim, num_heads=2)
        self.linear = nn.Linear(input_dim + output_dim, 256)
        self.final_linear = nn.Linear(256, 1)

    def forward(self, input):
        cross_out = self.cross1(input, input)
        cross_out = self.cross2(input, cross_out)
        cross_out = self.cross3(input, cross_out)
        cross_out = self.cross4(input, cross_out)
        mlp_out = self.mlp(input)
        cat_res = torch.cat((cross_out, mlp_out), dim=-1)
        output, _ = self.attention(cat_res, cat_res, cat_res)
        output = nn.functional.relu(self.linear(output))
        output = self.final_linear(output)

        return output

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    writer = SummaryWriter('./logs/')
    # raw = read_element(sort=True).values
    raw = read_pro_features().values
    features = raw[:, :-1]
    target = raw[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.4)
    print(x_train.shape)
    print(x_test.shape)

    if torch.cuda.is_available():
        x_train = torch.from_numpy(x_train).to(device)
        x_test = torch.from_numpy(x_test).to(device)
        y_train = torch.from_numpy(y_train).to(device)
        y_test = torch.from_numpy(y_test).to(device)
    else:
        x_train = torch.from_numpy(x_train)
        x_test = torch.from_numpy(x_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)

    model = cross_attention(input_dim=6)
    if torch.cuda.is_available():
        model.to(device)
    model.double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch_num = 50000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            out = model(x_train)
            loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
            writer.add_scalar('Loss/train', loss.data.item(), epoch)
            # print('loss:', loss.data.item())
            # loss_list.append(loss.data.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        if epoch % 10 == 9:
            print('epoch : ', epoch)
            pred = model(x_test)
            loss = criterion(torch.squeeze(pred), torch.squeeze(y_test))
            print('test loss:', loss.data.item())
            writer.add_scalar('Loss/test', loss.data.item(), epoch)
            if torch.cuda.is_available():
                r2 = r2_score(torch.squeeze(y_test.cpu()).detach().numpy(), torch.squeeze(pred.cpu()).detach().numpy())
                writer.add_scalar('R2', r2, epoch)
                print('r2:', r2)
            else:
                r2 = r2_score(torch.squeeze(y_test).detach().numpy(), torch.squeeze(pred).detach().numpy())
                writer.add_scalar('R2', r2, epoch)
                print('r2:', r2)
