import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


class MVN(nn.Module):
    def __init__(self, mean, var):
        super().__init__()
        self.reset(mean, var)

    def reset(self, mean, var):
        self.mean = nn.Parameter(mean)
        self.var = nn.Parameter(var)
        self.params = {'mean': mean, 'var': var}
        self.d = self.mean.size(-1)
        if self.d == 1:
            self.model = Normal(mean, var.sqrt())
        else:
            self.model = MultivariateNormal(mean, var)
        return

    def pdf(self, x):
        if self.d == 1:
            x = x.squeeze(-1)
        pdf_ = self.model.log_prob(x).exp()
        return pdf_

    def cdf(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x).to(self.mean.device)
        cdf_ = self.model.cdf(x)
        return cdf_

    def cdf_numpy(self, x):
        return self.cdf(x).cpu().numpy()

    def score(self, x):
        if self.d == 1:
            score_ = -1 * torch.matmul((x - self.mean), self.var ** (-1)).view(-1, 1)
        else:
            score_ = -1 * torch.matmul((x - self.mean), torch.linalg.inv(self.var))
        return score_

    def hscore(self, x):
        mean = self.mean
        if self.d == 1:
            invcov = self.var ** (-1)
            t1 = 0.5 * ((x - mean) * invcov * invcov).matmul((x - mean).transpose(-1, -2))
            t2 = - invcov
        else:
            invcov = torch.linalg.inv(self.var)
            t1 = 0.5 * (x - mean).matmul(invcov).matmul(invcov).matmul((x - mean).transpose(-1, -2))
            t2 = - invcov.diagonal().sum()
        t1 = t1.diagonal(dim1=-2, dim2=-1)
        hscore_ = t1 + t2
        return hscore_

    def fit(self, x):
        with torch.no_grad():
            mean = x.mean(dim=0)
            centered_x = x - mean
            var = centered_x.t().matmul(centered_x) / centered_x.size(0)
            epsilon = 1e-6
            var += epsilon * torch.eye(centered_x.size(-1), device=x.device)
            if self.d == 1:
                var = var.view(-1)
            self.reset(mean, var)
        return


def mvn(params):
    mean = params['mean']
    var = params['var']
    model = MVN(mean, var)
    return model
