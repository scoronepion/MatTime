import torch
import torch.nn as nn
import torch.optim as optim
from predata import read_element
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(45, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 3)

    def forward(self, input):
        output = nn.functional.relu(self.layer1(input))
        output = nn.functional.relu(self.layer2(output))
        output = nn.functional.relu(self.layer3(output))

        return output

if __name__ == '__main__':
    raw, _ = read_element(noise=True)

    features = raw.iloc[:, :-3].values
    target = raw.iloc[:, -3:].values

    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.3)
    print(x_train.shape)
    print(x_test.shape)

    device = torch.device('cuda:0')
    x_train = torch.from_numpy(x_train).to(device)
    x_test = torch.from_numpy(x_test).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    y_test = torch.from_numpy(y_test).to(device)

    mlp = MLP()
    mlp.to(device)
    mlp.double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters())

    epoch_num = 1000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            out = mlp(x_train)
            loss = criterion(out, y_train)
            # print('loss:', loss.data.item())
            # loss_list.append(loss.data.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        print('epoch : ', epoch)
        pred = mlp(x_test)
        loss = criterion(pred, y_test)
        print('test loss:', loss.data.item())
        print('r2:', r2_score(y_test.cpu(), pred.data.cpu()))