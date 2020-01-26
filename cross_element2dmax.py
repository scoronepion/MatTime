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
        trans = torch.einsum('bij,j->bi', [x, self.w])
        x_trans = torch.einsum('bij,bi->bij', [x0, trans])
        result = x_trans + self.b + x

if __name__ == '__main__':
    raw = read_element()
    print(raw.info())
    print(raw.head())