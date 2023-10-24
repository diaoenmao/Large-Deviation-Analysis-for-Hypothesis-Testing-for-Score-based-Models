import copy
import numpy as np
import torch
import torch.nn as nn
import model
from config import cfg
from tqdm import tqdm


def stable_log_mean_exp(v):
    m = v.max()
    return m + (v - m).exp().mean().log()


class HST:
    def __init__(self):
        self.numerical_bound = 500
        self.optim_iter = 10
        super().__init__()

    def compute_fpr_tpr_theoretical(self, null, alter, null_model, alter_model, threshold):
        def fpr_objective(theta, T):
            exponent = theta * (self.score(null, null_model, alter_model) / 2)
            obj = stable_log_mean_exp(exponent) - theta * T
            return obj

        def fnr_objective(theta, T):
            exponent = theta * (-self.score(alter, null_model, alter_model) / 2)
            obj = stable_log_mean_exp(exponent) + theta * T
            return obj

        def fpr_closure():
            fpr_optimizer.zero_grad()
            loss = fpr_objective(logit_fpr_theta_i.sigmoid() * self.numerical_bound, threshold_i)
            loss.backward()
            return loss

        def fnr_closure():
            fnr_optimizer.zero_grad()
            loss = fnr_objective(logit_fnr_theta_i.sigmoid() * self.numerical_bound, threshold_i)
            loss.backward()
            return loss

        threshold[threshold == -float('inf')] = 10 * threshold[torch.isfinite(threshold)].min()
        threshold[threshold == float('inf')] = 10 * threshold[torch.isfinite(threshold)].max()
        fpr, fnr = [], []
        fpr_log_theta = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, device=null.device)) for _ in range(len(threshold))])
        fnr_log_theta = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, device=null.device)) for _ in range(len(threshold))])
        for i in range(len(threshold)):
            threshold_i = threshold[i]
            logit_fpr_theta_i = fpr_log_theta[i]
            logit_fnr_theta_i = fnr_log_theta[i]
            fpr_optimizer = torch.optim.LBFGS([logit_fpr_theta_i], lr=1)
            fnr_optimizer = torch.optim.LBFGS([logit_fnr_theta_i], lr=1)
            for _ in range(self.optim_iter):
                fpr_optimizer.step(fpr_closure)
                fnr_optimizer.step(fnr_closure)
            fpr_i = fpr_objective(logit_fpr_theta_i.sigmoid() * self.numerical_bound, threshold_i).exp()
            fnr_i = fnr_objective(logit_fnr_theta_i.sigmoid() * self.numerical_bound, threshold_i).exp()
            # print('fpr', logit_fpr_theta_i.data.sigmoid() * self.numerical_bound, fpr_i)
            # print('fnr', logit_fnr_theta_i.data.sigmoid() * self.numerical_bound, fnr_i)
            fpr.append(fpr_i.item())
            fnr.append(fnr_i.item())
        fpr = np.array(fpr)
        fnr = np.array(fnr)
        return fpr, fnr

    def score(self, data, null_model, alter_model):
        """Calculate Hyvarinen Score Difference"""
        score = 2 * (null_model.hscore(data) - alter_model.hscore(data))
        score = score.reshape(-1)
        return score
