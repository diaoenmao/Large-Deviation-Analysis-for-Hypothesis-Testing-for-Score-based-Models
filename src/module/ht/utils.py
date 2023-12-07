import numpy as np
import torch
import torch.nn as nn
from config import cfg


def make_score(data, null_model, alter_model, score_fn, n):
    batch_size = 100
    score = []
    total_batches = int(np.ceil(data.size(0) / batch_size))
    for i in range(total_batches):
        model_idx = i % len(alter_model)
        num_samples = n * batch_size
        indices = torch.randint(0, len(data), (num_samples,))
        data_i = data[indices]
        score_i = score_fn(data_i, null_model, alter_model[model_idx])
        score_i = score_i.view(n, -1).mean(dim=0)
        score.append(score_i)
    score = torch.cat(score)[:data.size(0)]
    return score


def compute_empirical(null, alter, null_model, alter_model, threshold, score_fn):
    with torch.no_grad():
        target = torch.cat([torch.zeros(null.size(0)), torch.ones(alter.size(0))], dim=0).to(null.device)
        null_score = make_score(null, null_model, alter_model, score_fn,
                                cfg['num_samples_test'])
        alter_score = make_score(alter, null_model, alter_model, score_fn,
                                 cfg['num_samples_test'])
        score = torch.cat([null_score, alter_score], dim=0)

        # Expand dimensions to allow broadcasting
        target = target[:, None].cpu().numpy()  # Shape [N, 1]
        score = score[:, None].cpu().numpy()  # Shape [N, 1]
        threshold = threshold.cpu().numpy()

        # Generate predictions based on thresholds
        pred = (score >= threshold).astype(float)    # Shape [N, T] where T is number of thresholds

        # Compute TP, TN, FP, FN
        TP = np.sum((pred == 1) & (target == 1), axis=0).astype(float)
        TN = np.sum((pred == 0) & (target == 0), axis=0).astype(float)
        FP = np.sum((pred == 1) & (target == 0), axis=0).astype(float)
        FN = np.sum((pred == 0) & (target == 1), axis=0).astype(float)

        # Compute FPR and FNR
        FPR = FP / (FP + TN)
        FNR = FN / (TP + FN)

    return FPR, FNR


def compute_theoretical(null, alter, null_model, alter_model, threshold, score_fn, optim_iter):
    def fpr_objective(theta, T):
        theta.data.clamp_(min=0)
        exponent = theta * (null_score - T)
        element_exp = exponent.exp()
        obj = element_exp.mean()
        return obj

    def fnr_objective(theta, T):
        theta.data.clamp_(min=0)
        exponent = theta * (-alter_score + T)
        element_exp = exponent.exp()
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

    with torch.no_grad():
        null_score = make_score(null, null_model, alter_model, score_fn, 1).detach()
        alter_score = make_score(alter, null_model, alter_model, score_fn, 1).detach()
    threshold[threshold == -float('inf')] = 1 * threshold[torch.isfinite(threshold)].min()
    threshold[threshold == float('inf')] = 1 * threshold[torch.isfinite(threshold)].max()
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
        for _ in range(optim_iter):
            fpr_optimizer.step(fpr_closure)
            fnr_optimizer.step(fnr_closure)
        fpr_i = fpr_objective(fpr_theta_i, threshold_i)
        fnr_i = fnr_objective(fnr_theta_i, threshold_i)
        fpr.append(fpr_i.item())
        fnr.append(fnr_i.item())
    fpr = np.array(fpr)
    fnr = np.array(fnr)
    return fpr, fnr
