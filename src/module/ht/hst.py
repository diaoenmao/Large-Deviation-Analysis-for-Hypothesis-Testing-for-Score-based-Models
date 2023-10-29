import copy
import numpy as np
import torch
import torch.nn as nn
import model
from config import cfg
from tqdm import tqdm


class HST:
    def __init__(self):
        self.optim_iter = 10
        super().__init__()

    def compute_fpr_tpr_theoretical(self, null, alter, null_model, alter_model, threshold):
        def fpr_objective(theta, T):
            theta.data.clamp_(min=0)
            exponent = theta * (self.score(null, null_model, alter_model) / 2 - T)
            element_exp = exponent.exp()
            element_exp.data.clamp_(min=0, max=1e3)
            obj = element_exp.mean()
            return obj

        def fnr_objective(theta, T):
            theta.data.clamp_(min=0)
            exponent = theta * (-self.score(alter, null_model, alter_model) / 2 + T)
            element_exp = exponent.exp()
            element_exp.data.clamp_(min=0, max=1e3)
            obj = element_exp.mean()
            return obj

        def fpr_closure():
            fpr_optimizer.zero_grad()
            obj = fpr_objective(fpr_theta_i, threshold_i)
            loss = obj
            loss.backward()
            return loss

        def fnr_closure():
            fnr_optimizer.zero_grad()
            obj = fnr_objective(fnr_theta_i, threshold_i)
            loss = obj
            loss.backward()
            return loss

        threshold[threshold == -float('inf')] = 10 * threshold[torch.isfinite(threshold)].min()
        threshold[threshold == float('inf')] = 10 * threshold[torch.isfinite(threshold)].max()
        fpr, fnr = [], []
        fpr_theta = nn.ParameterList(
            [nn.Parameter(torch.ones(1, device=null.device)) for _ in range(len(threshold))])
        fnr_theta = nn.ParameterList(
            [nn.Parameter(torch.ones(1, device=null.device)) for _ in range(len(threshold))])
        for i in range(len(threshold)):
            threshold_i = threshold[i]
            fpr_theta_i = fpr_theta[i]
            fnr_theta_i = fnr_theta[i]
            fpr_optimizer = torch.optim.LBFGS([fpr_theta_i], lr=1)
            fnr_optimizer = torch.optim.LBFGS([fnr_theta_i], lr=1)
            for _ in range(self.optim_iter):
                fpr_optimizer.step(fpr_closure)
                fnr_optimizer.step(fnr_closure)
            fpr_i = fpr_objective(fpr_theta_i, threshold_i)
            fnr_i = fnr_objective(fnr_theta_i, threshold_i)
            # print(i, 'fpr', fpr_theta_i.data, fpr_i)
            # print(i, 'fnr', fnr_theta_i.data, fnr_i)
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
