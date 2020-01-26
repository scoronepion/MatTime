import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from big_predata import read_element
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

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


class mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(mlp, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.linear2 = nn.Linear(256, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, input):
        output = nn.functional.relu(self.linear1(input))
        output = self.dropout(output)
        output = nn.functional.relu(self.linear2(output))
        output = self.dropout(output)
        output = nn.functional.relu(self.linear3(output))
        output = self.dropout(output)
        output = nn.functional.relu(self.linear4(output))

        return output

class cross_mlp(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cross_mlp, self).__init__()
        self.cross1 = cross_layer(input_dim)
        self.cross2 = cross_layer(input_dim)
        self.cross3 = cross_layer(input_dim)
        self.cross4 = cross_layer(input_dim)
        self.mlp = mlp(input_dim, output_dim)
        self.final_linear = nn.Linear(input_dim + output_dim, 1)

    def forward(self, input):
        cross_out = self.cross1(input, input)
        cross_out = self.cross2(input, cross_out)
        cross_out = self.cross3(input, cross_out)
        cross_out = self.cross4(input, cross_out)
        mlp_out = self.mlp(input)
        cat_res = torch.cat((cross_out, mlp_out), dim=-1)
        output = self.final_linear(cat_res)

        return output

if __name__ == '__main__':
    raw = read_element().values
    features = raw[:, :-1]
    target = raw[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.4)
    print(x_train.shape)
    print(x_test.shape)

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    model = cross_mlp(input_dim=56, output_dim=56)
    model.double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch_num = 20000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            out = model(x_train)
            loss = criterion(torch.squeeze(out), torch.squeeze(y_train))
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
            print('r2:', r2_score(torch.squeeze(y_test).detach().numpy(), torch.squeeze(pred).detach().numpy()))
            # print('weight: ', model.embedding.embedding.weight)

    # f.close()
