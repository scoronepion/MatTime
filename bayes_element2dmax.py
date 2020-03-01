import torch
import torch.nn as nn
import torch.optim as optim
import math
import numpy as np
from big_predata import read_element, read_pro_features, calc_pac
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

PI = 0.5
SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])

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

class Gaussian(nn.Module):
    def __init__(self, mu, rho):
        '''重参数技巧'''
        super(Gaussian, self).__init__()
        self.mu = mu
        self.rho = rho
        # 标准高斯分布
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        '''返回方差'''
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            epsilon = self.normal.sample(self.rho.size()).to(device)
        else:
            epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi)) 
               - torch.log(self.sigma) 
               - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class ScaleMixtureGaussian(nn.Module):
    def __init__(self, pi, sigma1, sigma2):
        super(ScaleMixtureGaussian, self).__init__()
        self.pi = pi
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            self.sigma1 = sigma1.to(device)
            self.sigma2 = sigma2.to(device)
        else:
            self.sigma1 = sigma1
            self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        # TODO: 为什么这里前面要加exp
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return torch.log(self.pi * prob1 + (1 - self.pi) * prob2).sum()

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 权重分布参数
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # 偏置分布参数
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5, -4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # 先验分布
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        # log 先验
        self.log_prior = 0
        # log 变分后验
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior = 0
            self.log_variational_posterior = 0

        return nn.functional.linear(input, weight, bias)

class BayesianNet(nn.Module):
    def __init__(self, features_dim):
        super(BayesianNet, self).__init__()
        # 注意力部分
        # self.attention = multi_heads_self_attention(feature_dim=features_dim, num_heads=3)

        self.linear1 = BayesianLinear(features_dim, 256)
        self.linear2 = BayesianLinear(256, 512)
        self.linear3 = BayesianLinear(512, 256)
        self.linear4 = BayesianLinear(256, 1)

    def forward(self, input, sample=False):
        # output, _ = self.attention(input, input, input)
        output = nn.functional.relu(self.linear1(input, sample=sample))
        output = nn.functional.relu(self.linear2(output, sample=sample))
        output = nn.functional.relu(self.linear3(output, sample=sample))
        output = self.linear4(output, sample=sample)

        return output

    def log_prior(self):
        return self.linear1.log_prior \
               + self.linear2.log_prior \
               + self.linear3.log_prior \
               + self.linear4.log_prior

    def log_variational_posterior(self):
        return self.linear1.log_variational_posterior \
               + self.linear2.log_variational_posterior \
               + self.linear3.log_variational_posterior \
               + self.linear4.log_variational_posterior
    
    def sample_elbo(self, input, target, samples_num=10):
        outputs = torch.zeros(samples_num, input.size()[0], 1)
        log_priors = torch.zeros(samples_num)
        log_variational_posteriors = torch.zeros(samples_num)
        for i in range(samples_num):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        # 计算负对数似然，-log p(D|w)，D 为训练数据，w 为学习的权重。这里可以看成网络输出（output）为均值的，任意设定值（可视为超参数）为方差的高斯分布？（存疑）
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            sigma = torch.tensor(np.exp(-3)).to(device)
            negative_log_likelihood = (-math.log(math.sqrt(2 * math.pi)) \
                                      - torch.log(sigma) \
                                      - ((outputs.cpu().mean(0) - target.cpu()) ** 2) / (2 * sigma ** 2)).sum()
        else:
            sigma = torch.tensor(np.exp(-3))
            negative_log_likelihood = (-math.log(math.sqrt(2 * math.pi)) \
                                    - torch.log(sigma) \
                                    - ((outputs.mean(0) - target) ** 2) / (2 * sigma ** 2)).sum()

        loss = -(log_variational_posterior - log_prior + negative_log_likelihood)
        
        return loss, log_prior, log_variational_posterior, negative_log_likelihood

def write_weight_histograms(writer, model, epoch):
    # 权重
    writer.add_histogram('histogram/w1_mu', model.linear1.weight_mu,epoch)
    writer.add_histogram('histogram/w1_rho', model.linear1.weight_rho,epoch)
    writer.add_histogram('histogram/w2_mu', model.linear2.weight_mu,epoch)
    writer.add_histogram('histogram/w2_rho', model.linear2.weight_rho,epoch)
    writer.add_histogram('histogram/w3_mu', model.linear3.weight_mu,epoch)
    writer.add_histogram('histogram/w3_rho', model.linear3.weight_rho,epoch)
    writer.add_histogram('histogram/w4_mu', model.linear4.weight_mu,epoch)
    writer.add_histogram('histogram/w4_rho', model.linear4.weight_rho,epoch)
    # 偏置
    writer.add_histogram('histogram/b1_mu', model.linear1.bias_mu,epoch)
    writer.add_histogram('histogram/b1_rho', model.linear1.bias_rho,epoch)
    writer.add_histogram('histogram/b2_mu', model.linear2.bias_mu,epoch)
    writer.add_histogram('histogram/b2_rho', model.linear2.bias_rho,epoch)
    writer.add_histogram('histogram/b3_mu', model.linear3.bias_mu,epoch)
    writer.add_histogram('histogram/b3_rho', model.linear3.bias_rho,epoch)
    writer.add_histogram('histogram/b4_mu', model.linear4.bias_mu,epoch)
    writer.add_histogram('histogram/b4_rho', model.linear4.bias_rho,epoch)

def write_loss_scalars(writer, epoch, loss, log_prior, log_variational_posterior, negative_log_likelihood):
    writer.add_scalar('logs/loss', loss, epoch)
    writer.add_scalar('logs/log_prior', log_prior, epoch)
    writer.add_scalar('logs/log_variational_posterior', log_variational_posterior, epoch)
    writer.add_scalar('logs/negative_log_likelihood', negative_log_likelihood, epoch)

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    writer = SummaryWriter('./logs/')
    raw = read_element(sort=True).values
    # raw = calc_pac(num=50)
    features = raw[:-1, :-1]
    target = raw[:-1, -1:]
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.1)
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

    model = BayesianNet(features_dim=45)
    if torch.cuda.is_available():
        model.to(device)
    model.double()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epoch_num = 50000

    for epoch in range(epoch_num):
        def closure():
            model.train()
            optimizer.zero_grad()
            write_weight_histograms(writer, model, epoch)
            loss, log_prior, log_variational_posterior, negative_log_likelihood = model.sample_elbo(x_train, y_train)
            loss.backward()
            write_loss_scalars(writer, epoch, loss, log_prior, log_variational_posterior, negative_log_likelihood)
            return loss

        optimizer.step(closure)

        if epoch % 10 == 9:
            print('epoch : ', epoch)
            model.eval()
            ensemble_size = 10
            pred = 0
            for i in range(10):
                # 权值分布采样网络结果，模拟 ensemble learning
                pred += model(x_test, sample=True)
            # 最后一个元素记录全职分布均值网络结果
            pred += model(x_test, sample=False)
            pred /= 10
            if torch.cuda.is_available():
                r2 = r2_score(torch.squeeze(y_test.cpu()).detach().numpy(), torch.squeeze(pred.cpu()).detach().numpy())
                writer.add_scalar('R2', r2, epoch)
                print('r2:', r2)
            else:
                r2 = r2_score(torch.squeeze(y_test).detach().numpy(), torch.squeeze(pred).detach().numpy())
                writer.add_scalar('R2', r2, epoch)
                print('r2:', r2)