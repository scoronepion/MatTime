import torch
import math

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