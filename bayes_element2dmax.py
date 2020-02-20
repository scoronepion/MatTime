import torch
import torch.nn as nn
import math
import numpy as np

PI = 0.5
SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])

class Gaussian(object):
    def __init__(self, mu, rho):
        '''重参数技巧'''
        super().__init__()
        self.mu = mu
        self.rho = rho
        # 标准高斯分布
        self.normal = torch.distributions.normal(0, 1)

    @property
    def sigma(self):
        '''返回方差'''
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi)) 
               - torch.log(self.sigma) 
               - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
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
        self.linear1 = BayesianLinear(features_dim, 256)
        self.linear2 = BayesianLinear(256, 512)
        self.linear3 = BayesianLinear(512, 256)
        self.linear4 = BayesianLinear(256, 1)

    def forward(self, input, sample=False):
        output = nn.functional.relu(self.linear1(input, sample=sample))
        output = nn.functional.relu(self.linear2(output, sample=sample))
        output = nn.functional.relu(self.linear3(output, sample=sample))
        output = self.linear4(output, sample=sample)

        return output

    def log_prior(self):
        return self.linear1.log_prior
               + self.linear2.log_prior
               + self.linear3.log_prior
               + self.linear4.log_prior

    def log_variational_posterior(self):
        return self.linear1.log_variational_posterior
               + self.linear2.log_variational_posterior
               + self.linear3.log_variational_posterior
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
        sigma = float(np.exp(-3))
        negative_log_likelihood = (-math.log(math.sqrt(2 * math.pi)) 
                                  - torch.log(sigma) 
                                  - ((outputs.mean(0) - target) ** 2) / (2 * sigma ** 2)).sum()

        loss = log_variational_posterior - log_prior + negative_log_likelihood
        
        return loss