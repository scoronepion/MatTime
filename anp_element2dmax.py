import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import collections
from big_predata import read_element
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

ANPRegressionDescription = collections.namedtuple(
    "ANPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points", "num_target_points"))

def preprocess():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    raw = read_element().values
    print("raw shape = " + str(raw.shape))

    # 训练集：除开最后三条，其余正常按照 target 和 context 划分
    features = raw[:-3, :-1]
    target = raw[:-3, -1:]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.4)
    print("training x_train shape = " + str(x_train.shape))
    print("training x_test shape = " + str(x_test.shape))
    if torch.cuda.is_available():
        # reshape to [B, context_observation, d_x/d_y]
        x_train = torch.from_numpy(x_train).unsqueeze(0).to(device)
        y_train = torch.from_numpy(y_train).unsqueeze(0).to(device)
        # reshape to [B, target_observation, d_x/d_y]
        x_test = torch.from_numpy(x_test).unsqueeze(0).to(device)
        y_test = torch.from_numpy(y_test).unsqueeze(0).to(device)
    else:
        # reshape to [B, context_observation, d_x/d_y]
        x_train = torch.from_numpy(x_train).unsqueeze(0)
        y_train = torch.from_numpy(y_train).unsqueeze(0)
        # reshape to [B, target_observation, d_x/d_y]
        x_test = torch.from_numpy(x_test).unsqueeze(0)
        y_test = torch.from_numpy(y_test).unsqueeze(0)

    query_train = ((x_train, y_train), x_test)
    train_set = ANPRegressionDescription(
        query=query_train,
        target_y=y_test,
        num_total_points=raw.shape[0],
        num_context_points=x_train.shape[1],
        num_target_points=x_test.shape[1]
    )

    # 测试集：最后三条用作 target，剩余的用作 context
    x_train = raw[:-3, :-1]
    y_train = raw[:-3, -1:]
    x_test = raw[-3:, :-1]
    y_test = raw[-3:, -1:]
    print("test x_train shape = " + str(x_train.shape))
    print("test x_test shape = " + str(x_test.shape))
    if torch.cuda.is_available():
        # reshape to [B, context_observation, d_x/d_y]
        x_train = torch.from_numpy(x_train).unsqueeze(0).to(device)
        y_train = torch.from_numpy(y_train).unsqueeze(0).to(device)
        # reshape to [B, target_observation, d_x/d_y]
        x_test = torch.from_numpy(x_test).unsqueeze(0).to(device)
        y_test = torch.from_numpy(y_test).unsqueeze(0).to(device)
    else:
        # reshape to [B, context_observation, d_x/d_y]
        x_train = torch.from_numpy(x_train).unsqueeze(0)
        y_train = torch.from_numpy(y_train).unsqueeze(0)
        # reshape to [B, target_observation, d_x/d_y]
        x_test = torch.from_numpy(x_test).unsqueeze(0)
        y_test = torch.from_numpy(y_test).unsqueeze(0)

    query_test = ((x_train, y_train), x_test)
    test_set = ANPRegressionDescription(
        query=query_test,
        target_y=y_test,
        num_total_points=raw.shape[0],
        num_context_points=x_train.shape[1],
        num_target_points=x_test.shape[1]
    )

    return train_set, test_set
    
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
    def __init__(self, feature_dim, num_heads=2, dropout=0.0):
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
        # # 为了 de_encoder 中后接 cross attention 能正常运算，此处需要线性层减少 1 维
        # self.linear = nn.Linear(feature_dim, 56)

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
        output = torch.squeeze(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        # pass through linear
        # output = self.linear(output)

        return output, attention

class DeterministicEncoder(nn.Module):
    def __init__(self):
        super(DeterministicEncoder, self).__init__()
        self.self_attention = multi_heads_self_attention(feature_dim=57, num_heads=3)
        self.cross_attention = multi_heads_self_attention(feature_dim=56, num_heads=2)
        # 连接 self-attention 与 cross-attention
        self.linear = nn.Linear(57, 56)

    def forward(self, x_context, y_context, x_target):
        '''
        x_context: [B, context_observations, d_x]
        y_context: [B, context_observations, d_y]
        x_target: [B, target_observations, d_y]
        return: [B,target_observations,d]
        '''
        # concat x and y
        encoder_input = torch.cat((x_context, y_context), dim=-1)
        # pass through self-attention
        # 为了下一层 cross attention 能正常运算，此处 self-attention 输出应该减少 1 维（变为 56 维）
        output, _ = self.self_attention(encoder_input, encoder_input, encoder_input)
        output = self.linear(output)
        # pass through cross-attention
        output, _ = self.cross_attention(x_context, output, x_target)

        return output

class LatentEncoder(nn.Module):
    def __init__(self, linear_feature_dim, hidden_size):
        super(LatentEncoder, self).__init__()
        self.self_attention = multi_heads_self_attention(feature_dim=57, num_heads=3)
        self.linear_relu = nn.Linear(linear_feature_dim, hidden_size * 2)
        self.linear_mean = nn.Linear(hidden_size * 2, hidden_size)
        self.linear_std = nn.Linear(hidden_size * 2, hidden_size)
    
    def forward(self, x, y):
        '''
        x_context: [B, context_observations, d_x]
        y_context: [B, context_observations, d_y]
        '''
        # concat x and y
        encoder_input = torch.cat((x, y), dim=-1)
        # pass through self-attention
        output, _ = self.self_attention(encoder_input, encoder_input, encoder_input)
        # mean
        output = torch.mean(output, dim=1)
        # pass through relu layer
        output = nn.functional.relu(self.linear_relu(output))
        # calculate mean and std
        mu = self.linear_mean(output)
        log_sigma = self.linear_std(output)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        return torch.distributions.normal.Normal(loc=mu, scale=sigma)

class Decoder(nn.Module):
    def __init__(self, feature_dim, y_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(feature_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, feature_dim)
        self.linear3 = nn.Linear(feature_dim, feature_dim)
        # self.linear4 = nn.Linear(feature_dim, y_dim * 2)
        # self.attention = multi_heads_self_attention(feature_dim=feature_dim)
        # # attention 输出
        # self.linear_attention = nn.Linear(feature_dim, feature_dim)
        self.linear_mu = nn.Linear(feature_dim, 1)
        self.linear_sigma = nn.Linear(feature_dim, 1)
        self.dropout = nn.Dropout(0.2)
        # self.layer_norm = nn.LayerNorm(feature_dim)

    def forward(self, context_rep, x_target):
        '''
            context_rep: context representation for target predictions, [B, target_observations, ?] (这里是 128+56)
            x_target: [B, target_observations, d_x]
        return: 
            dist: A multivariate Gaussian over the target points, [B,target_observations,d_y]
            mu: [B,target_observations,d_x]
            sigma: [B,target_observations,d_x]
        '''
        # concat context representation and x_target
        output = torch.cat((context_rep, x_target), dim=-1)

        # pass through mlp
        output = nn.functional.relu(self.linear1(output))
        output = self.dropout(output)
        output = nn.functional.relu(self.linear2(output))
        output = self.dropout(output)
        output = nn.functional.relu(self.linear3(output))
        # output = self.layer_norm(output)
        # output = self.linear4(output)

        # output, _ = self.attention(output, output, output)
        # output = nn.functional.relu(self.linear_attention(output))
        # get mean and std
        # (split_size_or_sections: size of a single chunk or list of sizes for each chunk)
        # mu, log_sigma = torch.split(output, split_size_or_sections=1, dim=-1)
        mu = self.linear_mu(output)
        log_sigma = self.linear_sigma(output)
        # mu = torch.squeeze(mu)
        # log_sigma = torch.squeeze(log_sigma, dim=0)
        sigma = 0.1 + 0.9 * nn.functional.softplus(log_sigma)
        # get the distribution
        # dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, scale_tril=sigma)
        dist = torch.distributions.normal.Normal(loc=mu, scale=sigma)

        return dist, mu, sigma

class ANP(nn.Module):
    def __init__(self, encoder_feature_dim, hidden_size, decoder_feature_dim, y_dim):
        super(ANP, self).__init__()
        self.de_encoder = DeterministicEncoder()
        self.la_encoder = LatentEncoder(encoder_feature_dim, hidden_size)
        self.decoder = Decoder(decoder_feature_dim, y_dim)

    def forward(self, query, num_targets, y_target=None):
        (x_context, y_context), x_target = query
        # pass through deterministic encoder
        # [B, num_targets, 56]
        de_rep = self.de_encoder(x_context, y_context, x_target)
        # pass through latentencoder
        prior = self.la_encoder(x_context, y_context)
        # 测试时，y_target 不存在，需要从 context 先验中采样
        if y_target is None:
            la_rep = prior.sample()
        else:
            # 训练时，y_target 存在，故将 target 作为后验
            posterior = self.la_encoder(x_target, y_target)
            la_rep = posterior.sample()
        # TODO: why?
        # [B, num_targets, hidden_size]
        la_rep = torch.unsqueeze(la_rep, 1).repeat(1, num_targets, 1)
        # concat representation
        representation = torch.cat((de_rep, la_rep), dim=-1)

        dist, mu, sigma = self.decoder(representation, x_target)

        # 训练时，y_target 存在，利用其计算 log_prob
        # 个人理解：模型生成了一个多元高斯分布，需要拿真实的 y 在该分布中取条件概率，值越大则表示生成的分布越准确
        if y_target is not None:
            log_p = dist.log_prob(y_target)
            posterior = self.la_encoder(x_target, y_target)
            kl = torch.distributions.kl.kl_divergence(posterior, prior).sum(dim=-1, keepdim=True)
            kl = kl.repeat(1, num_targets)
            loss = - torch.mean(log_p - kl / float(num_targets))
        else:
            # 预测时，返回 None
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss

if __name__ == '__main__':
    train_dataset, test_dataset = preprocess()
    writer = SummaryWriter('./logs/')
    model = ANP(encoder_feature_dim=57, hidden_size=128, decoder_feature_dim=128+56+56, y_dim=1)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model.to(device)
    model.double()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch_num = 100000

    for epoch in range(epoch_num):
        def closure():
            optimizer.zero_grad()
            _, _, log_p, _, loss = model(train_dataset.query, train_dataset.num_target_points, train_dataset.target_y)
            writer.add_scalar('Loss/train', loss.data.item(), epoch)
            # print(loss.data.item())
            loss.backward()
            return loss
        
        optimizer.step(closure)

        if epoch % 10 == 9:
            print('epoch : ', epoch)
            mu, sigma, _, _, loss = model(test_dataset.query, test_dataset.num_target_points, test_dataset.target_y)
            writer.add_scalar('Loss/test', loss.data.item(), epoch)

            writer.add_scalar('mean/-3', torch.squeeze(mu.data).cpu().numpy()[0], epoch)
            writer.add_scalar('mean/-2', torch.squeeze(mu.data).cpu().numpy()[1], epoch)
            writer.add_scalar('mean/-1', torch.squeeze(mu.data).cpu().numpy()[2], epoch)

            writer.add_scalar('std/-3', torch.squeeze(sigma.data).cpu().numpy()[0], epoch)
            writer.add_scalar('std/-2', torch.squeeze(sigma.data).cpu().numpy()[1], epoch)
            writer.add_scalar('std/-1', torch.squeeze(sigma.data).cpu().numpy()[2], epoch)

            print('test loss:', loss.data.item())
            print("test mean:", torch.squeeze(mu.data))
            print("test std:", torch.squeeze(sigma.data))
            # print('r2:', r2_score(torch.squeeze(y_test).detach().numpy(), torch.squeeze(pred).detach().numpy()))
            # print('weight: ', model.embedding.embedding.weight)
