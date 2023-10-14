import copy
import numpy as np
import torch
import torch.nn as nn
import model
from config import cfg


def stable_log_mean_exp(v):
    m = v.max()
    return m + (v - m).exp().mean().log()


class HST:
    def __init__(self):
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
            loss = fpr_objective(log_fpr_theta_i.exp(), threshold_i)
            loss.backward()
            return loss

        def fnr_closure():
            fnr_optimizer.zero_grad()
            loss = fnr_objective(log_fnr_theta_i.exp(), threshold_i)
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
            log_fpr_theta_i = fpr_log_theta[i]
            log_fnr_theta_i = fnr_log_theta[i]
            fpr_optimizer = torch.optim.Adam([log_fpr_theta_i], lr=1)
            fnr_optimizer = torch.optim.Adam([log_fnr_theta_i], lr=1)
            for _ in range(10):
                fpr_optimizer.step(fpr_closure)
                fnr_optimizer.step(fnr_closure)
            fpr_i = (fpr_objective(log_fpr_theta_i.exp(), threshold_i)).exp().item()
            fnr_i = (fnr_objective(log_fnr_theta_i.exp(), threshold_i)).exp().item()
            print(log_fpr_theta_i.exp(), log_fnr_theta_i.exp(), fpr_i, fnr_i)
            fpr.append(fpr_i)
            fnr.append(fnr_i)
        print(fpr)
        print(fnr)
        exit()

        return

    def score(self, data, null_model, alter_model):
        """Calculate Hyvarinen Score Difference"""
        score = 2 * (null_model.hscore(data) - alter_model.hscore(data))
        score = score.reshape(-1)
        return score
