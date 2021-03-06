import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from big_predata import read_element, read_over_element, read_cmp
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

class chemical_embedding(nn.Module):
    '''返回元素 embedding 表示'''
    def __init__(self, length, embedding_size):
        '''length: 元素数量；embedding_size: 嵌入大小'''
        super(chemical_embedding, self).__init__()
        self.length = length
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(length, embedding_size)

    def forward(self, input):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        index = np.tile([i for i in range(self.length)], (input.size(0), 1))
        if torch.cuda.is_available():
            index = torch.tensor(index, dtype=torch.long).to(device)
        else:
            index = torch.tensor(index, dtype=torch.long)
        embed = self.embedding(index).view(-1)
        # 输入变换
        if torch.cuda.is_available():
            trans = torch.zeros(self.length, self.length * self.embedding_size, dtype=torch.float64).to(device)
        else:
            trans = torch.zeros(self.length, self.length * self.embedding_size, dtype=torch.float64)
        flag = 0
        for i in range(self.length):
            for j in range(self.embedding_size):
                trans[i][flag + j] = 1
            flag += self.embedding_size
        expanded_data = torch.einsum('ij,jk->ik', [input, trans]).view(-1)
        result = (expanded_data * embed).view(input.size(0), 1, -1)

        return result

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
        # output = torch.squeeze(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        # # pass through linear
        # output = nn.functional.relu(self.linear_1(output))
        # output = nn.functional.relu(self.linear_2(output))

        # # pass through layer final
        # output = self.layer_final(output)

        return output, attention

class embedding_mlp(nn.Module):
    def __init__(self, length, embedding_size):
        super(embedding_mlp, self).__init__()
        self.embedding = chemical_embedding(length=length, embedding_size=embedding_size)
        # self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(length * embedding_size, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linera3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 1)

    def forward(self, input):
        embed = self.embedding(input)
        output = nn.functional.relu(self.linear1(embed))
        # output = self.dropout(output)
        output = nn.functional.relu(self.linear2(output))
        # output = self.dropout(output)
        output = nn.functional.relu(self.linera3(output))
        output = self.linear4(output)

        return output

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

class embedding_attention_attention(nn.Module):
    def __init__(self, length, embedding_size):
        super(embedding_attention_attention, self).__init__()
        self.embedding = chemical_embedding(length=length, embedding_size=embedding_size)
        self.attention1 = multi_heads_self_attention(feature_dim=length * embedding_size, num_heads=2)
        self.attention2 = multi_heads_self_attention(feature_dim=length * embedding_size, num_heads=2)
        self.linear_final = nn.Linear(length * embedding_size, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        embed = self.embedding(input)
        output, _ = self.attention1(embed, embed, embed)
        output, _ = self.attention2(output, output, output)
        output = self.linear_final(output)

        return output

class embedding_attention_mlp(nn.Module):
    def __init__(self, length, embedding_size):
        super(embedding_attention_mlp, self).__init__()
        self.embedding = chemical_embedding(length=length, embedding_size=embedding_size)
        self.attention = multi_heads_self_attention(feature_dim=length * embedding_size, num_heads=9)
        self.layer_norm = nn.LayerNorm(length * embedding_size)
        self.linear1 = nn.Linear(length * embedding_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear_final = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        embed = self.embedding(input)
        output, _ = self.attention(embed, embed, embed)
        output = self.layer_norm(output)
        output = nn.functional.relu(self.linear1(output))
        output = self.dropout(output)
        output = nn.functional.relu(self.linear2(output))
        output = self.dropout(output)
        output = nn.functional.relu(self.linear3(output))
        output = self.linear_final(output)

        return output

class pure_embedding(nn.Module):
    def __init__(self, length, embedding_size):
        super(pure_embedding, self).__init__()
        self.embedding = chemical_embedding(length=length, embedding_size=embedding_size)
        self.linear = nn.Linear(length * embedding_size, 512)
        self.linear_final = nn.Linear(512, 1)

    def forward(self, input):
        embed = self.embedding(input)
        output = nn.functional.relu(self.linear(embed))
        output = self.linear_final(output)

        return output

class embedding_attention_cross(nn.Module):
    def __init__(self, length, embedding_size):
        super(embedding_attention_cross, self).__init__()
        self.embedding = chemical_embedding(length=length, embedding_size=embedding_size)
        self.attention = multi_heads_self_attention(feature_dim=length * embedding_size, num_heads=2)
        self.cross1 = cross_layer(length * embedding_size)
        self.cross2 = cross_layer(length * embedding_size)
        self.cross3 = cross_layer(length * embedding_size)
        self.cross4 = cross_layer(length * embedding_size)
        self.linear_final = nn.Linear(length * embedding_size, 1)

    def forward(self, input):
        embed = self.embedding(input)
        att_out, _ = self.attention(embed, embed, embed)
        att_out = torch.squeeze(att_out)
        output = self.cross1(att_out, att_out)
        output = self.cross2(att_out, output)
        output = self.cross3(att_out, output)
        output = self.cross4(att_out, output)
        output = self.linear_final(output)

        return output


if __name__ == '__main__':
    # f = open('0122.log', 'a')
    # sys.stdout = f
    # sys.stderr = f
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    writer = SummaryWriter('./logs/')
    # raw = read_element(sort=True).values
    raw = read_cmp().values
    # raw = np.expand_dims(raw, axis=1)

    # # 最后三条作为展示集
    # show_features = raw[-3:, :-1]
    # show_target = raw[-3:, -1:]

    features = raw[:-1, :-1]
    target = raw[:-1, -1:]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1)
    print(x_train.shape)
    print(x_test.shape)

    # scale for loss
    scaled_vector = y_train.copy()
    sum_list = [0] * 5
    for i in range(scaled_vector.shape[0]):
        if 0.2 <= scaled_vector[i] <= 1.5:
            sum_list[0] += 1
            scaled_vector[i] = 1
        elif 2 <= scaled_vector[i] <= 2.8:
            sum_list[1] += 1
            scaled_vector[i] = 2
        elif 3 <= scaled_vector[i] <= 4.5:
            sum_list[2] += 1
            scaled_vector[i] = 3
        elif 5 <= scaled_vector[i] <= 8:
            sum_list[3] += 1
            scaled_vector[i] = 4
        else:
            sum_list[4] += 1
            scaled_vector[i] = 5
    scale_list = [0] * 5
    for i in range(5):
        scale_list[i] = float(sum(sum_list)) / float(sum_list[i])
    for i in range(scaled_vector.shape[0]):
        if scaled_vector[i] == 1:
            scaled_vector[i] = scale_list[0]
        elif scaled_vector[i] == 2:
            scaled_vector[i] = scale_list[1]
        elif scaled_vector[i] == 3:
            scaled_vector[i] = scale_list[2]
        elif scaled_vector[i] == 4:
            scaled_vector[i] = scale_list[3]
        else:
            scaled_vector[i] = scale_list[4]

    # print(scaled_vector)

    if torch.cuda.is_available():
        x_train = torch.from_numpy(x_train).to(device)
        x_test = torch.from_numpy(x_test).to(device)
        y_train = torch.from_numpy(y_train).to(device)
        y_test = torch.from_numpy(y_test).to(device)
        scaled_vector = torch.from_numpy(scaled_vector).to(device)
        # show_features = torch.from_numpy(show_features).to(device)
        # show_target = torch.from_numpy(show_target).to(device)
    else:
        x_train = torch.from_numpy(x_train)
        x_test = torch.from_numpy(x_test)
        y_train = torch.from_numpy(y_train)
        y_test = torch.from_numpy(y_test)
        scaled_vector = torch.from_numpy(scaled_vector)
        # show_features = torch.from_numpy(show_features)
        # show_target = torch.from_numpy(show_target)

    model = embedding_attention(length=45, embedding_size=9)
    if torch.cuda.is_available():
        model.to(device)
    model.double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # # ensemble models
    # if torch.cuda.is_available():
    #     models = [embedding_attention(length=45, embedding_size=9).to(device).double() for i in range(10)]
    # else:
    #     models = [embedding_attention(length=45, embedding_size=9).double() for i in range(10)]
    # criterion = nn.MSELoss()
    # optimizer = optim.Adam([{"params": item.parameters()} for item in models], lr=0.0001)

    epoch_num = 50000
    save_flag = False

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            out = model(x_train)
            loss = (scaled_vector * (torch.squeeze(out) - torch.squeeze(y_train)) ** 2).mean()
            # loss = torch.mean((torch.squeeze(out) - torch.squeeze(y_train)) ** 2)
            # loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
            writer.add_scalar('Loss/train', loss.data.item(), epoch)
            # print('loss:', loss.data.item())
            # loss_list.append(loss.data.item())
            loss.backward()
            return loss

        # ensemble way
        # def closure():
        #     optimizer.zero_grad()
        #     for item in models:
        #         out = item(x_train)
        #         loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
        #         loss.backward()

        optimizer.step(closure)

        if epoch % 10 == 9:
            print('epoch : ', epoch)
            pred = model(x_test)
            # show_pred = model(show_features)
            # loss = criterion(torch.squeeze(pred), torch.squeeze(y_test))
            loss = (scaled_vector * (torch.squeeze(pred) - torch.squeeze(y_test)) ** 2).mean()
            # loss = torch.mean((torch.squeeze(pred) - torch.squeeze(y_test)) ** 2)
            writer.add_scalar('Loss/test', loss.data.item(), epoch)
            print('test loss:', loss.data.item())
            if torch.cuda.is_available():
                r2 = r2_score(torch.squeeze(y_test.cpu()).detach().numpy(), torch.squeeze(pred.cpu()).detach().numpy())
                writer.add_scalar('R2', r2, epoch)
                print('r2:', r2)
                if r2 > 0.80:
                    save_flag = True
                # show_case_result = torch.squeeze(show_pred.cpu()).detach().numpy()
                # print('show case:', show_case_result)
                # writer.add_scalar('pred/-3', show_case_result[-3], epoch)
                # writer.add_scalar('pred/-2', show_case_result[-2], epoch)
                # writer.add_scalar('pred/-1', show_case_result[-1], epoch)
            else:
                r2 = r2_score(torch.squeeze(y_test).detach().numpy(), torch.squeeze(pred).detach().numpy())
                writer.add_scalar('R2', r2, epoch)
                print('r2:', r2)
                if r2 > 0.80:
                    save_flag = True
                # show_case_result = torch.squeeze(show_pred).detach().numpy()
                # print('show case:', show_case_result)
                # writer.add_scalar('pred/-3', show_case_result[-3], epoch)
                # writer.add_scalar('pred/-2', show_case_result[-2], epoch)
                # writer.add_scalar('pred/-1', show_case_result[-1], epoch)
            # print('weight: ', model.embedding.embedding.weight)

        if save_flag:
            torch.save(model, './models/embedding_attention_Full_Dmax_no75_scaled_080.bin')
            print('model save succeed')
            break

            # ensemble way
            # loss_list = []
            # r2_list = []
            # pred_res = 0
            # for item in models:
            #     pred = item(x_test)
            #     loss = criterion(torch.squeeze(pred), torch.squeeze(y_test))
            #     loss_list.append(loss.data.item())

            #     if torch.cuda.is_available():
            #         # r2 = r2_score(torch.squeeze(y_test.cpu()).detach().numpy(), torch.squeeze(pred.cpu()).detach().numpy())
            #         # r2_list.append(r2)
            #         pred_res += torch.squeeze(pred.cpu()).detach().numpy()
            #     else:
            #         # r2 = r2_score(torch.squeeze(y_test).detach().numpy(), torch.squeeze(pred).detach().numpy())
            #         # r2_list.append(r2)
            #         pred_res += torch.squeeze(pred).detach().numpy()

            # pred_res /= float(len(models))
            # if torch.cuda.is_available():
            #     r2 = r2_score(torch.squeeze(y_test.cpu()).detach().numpy(), pred_res)
            # else:
            #     r2 = r2_score(torch.squeeze(y_test).detach().numpy(), pred_res)
            # print('loss = ', loss_list)
            # print('r2 = ', r2)
            # writer.add_scalar('R2', r2, epoch)