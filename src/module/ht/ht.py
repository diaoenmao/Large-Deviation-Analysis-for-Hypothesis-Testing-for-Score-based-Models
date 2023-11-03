import copy
import torch
import numpy as np
import model
from config import cfg
from .lrt import LRT
from .hst import HST
from sklearn.metrics import roc_curve


class HypothesisTest:
    def __init__(self, ht_mode, num_samples_emp):
        self.ht_mode = ht_mode.split('-')
        self.num_samples_emp = num_samples_emp
        self.ht = self.make_ht()
        self.result = {'threshold': [], 'fpr': [], 'fnr': []}
        self.num_threshold = 300
        self.num_test_emp = 10
        self.batch_size = 100

    def make_ht(self):
        if self.ht_mode[0] in ['lrt']:
            ht = LRT()
        elif self.ht_mode[0] in ['hst']:
            ht = HST()
        else:
            raise ValueError('Not valid ht mode')
        return ht

    def make_score(self, null, alter, null_model, alter_model):
        null_score = []
        alter_score = []
        for i in range(null.size(0) // self.batch_size):
            null_i = bootstrap(null, cfg['num_samples_test'] * self.batch_size)
            alter_i = bootstrap(alter, cfg['num_samples_test'] * self.batch_size)
            null_score_i = self.ht.score(null_i, null_model, alter_model)
            alter_score_i = self.ht.score(alter_i, null_model, alter_model)
            null_score_i = null_score_i.view(cfg['num_samples_test'], -1).mean(dim=0)
            alter_score_i = alter_score_i.view(cfg['num_samples_test'], -1).mean(dim=0)
            null_score.append(null_score_i)
            alter_score.append(alter_score_i)
        null_score = torch.cat(null_score)
        alter_score = torch.cat(alter_score)
        score = torch.cat([null_score, alter_score], dim=0)
        return score

    def make_threshold(self, null, alter, null_model, alter_model):
        with torch.no_grad():
            if not isinstance(alter_model, list):
                score = self.make_score(null, alter, null_model, alter_model)
                target = torch.cat([torch.zeros(null.size(0)), torch.ones(alter.size(0))], dim=0).to(null.device)
                fpr, _, threshold = roc_curve(target.cpu().numpy(), score.cpu().numpy())
                idx = np.linspace(0, 1, self.num_threshold)
                threshold = np.interp(idx, fpr, threshold)
                threshold = torch.tensor(threshold, device=null.device, dtype=torch.float32)
            else:
                target = torch.cat([torch.zeros(null.size(0)), torch.ones(alter.size(0))], dim=0).to(null.device)
                idx = np.linspace(0, 1, self.num_threshold)
                threshold = []
                for i in range(len(alter_model)):
                    score_i = self.make_score(null, alter, null_model, alter_model[i])
                    fpr_i, _, threshold_i = roc_curve(target.cpu().numpy(), score_i.cpu().numpy(),
                                                      drop_intermediate=False)
                    threshold_i = np.interp(idx, fpr_i, threshold_i)
                    threshold_i = torch.tensor(threshold_i, device=null.device, dtype=torch.float32)
                    threshold.append(threshold_i)
                threshold = torch.stack(threshold, dim=0).mean(dim=0)
        return threshold

    def test(self, input):
        null, alter, null_param, alter_param = input['null'], input['alter'], input['null_param'], input['alter_param']
        null_model = eval('model.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
        if self.num_samples_emp is not None:
            alter_model = []
            alter_split = torch.split(alter, self.num_samples_emp, dim=0)
            for i in range(self.num_test_emp):
                alter_model_i = eval('model.{}(copy.deepcopy(null_model.params)).to(cfg["device"])'.format(
                    cfg['model_name']))
                alter_model_i.fit(alter_split[i])
                alter_model.append(alter_model_i)
        else:
            alter_model = eval('model.{}(alter_param).to(cfg["device"])'.format(cfg['model_name']))
        threshold = self.make_threshold(null, alter, null_model, alter_model)
        if self.ht_mode[0] in ['lrt', 'hst']:
            if self.ht_mode[1] == 'e':
                with torch.no_grad():
                    target = torch.cat([torch.zeros(null.size(0)), torch.ones(alter.size(0))], dim=0).to(null.device)
                    if not isinstance(alter_model, list):
                        score = self.make_score(null, alter, null_model, alter_model)
                        fpr, fnr = compute_fpr_tpr_empirical(target, score, threshold)
                    else:
                        fpr, fnr = [], []
                        for i in range(len(alter_model)):
                            score_i = self.make_score(null, alter, null_model, alter_model[i])
                            fpr_i, fnr_i = compute_fpr_tpr_empirical(target, score_i, threshold)
                            fpr.append(fpr_i)
                            fnr.append(fnr_i)
                        fpr = np.stack(fpr, axis=0).mean(axis=0)
                        fnr = np.stack(fnr, axis=0).mean(axis=0)
            elif self.ht_mode[1] == 't':
                if not isinstance(alter_model, list):
                    fpr, fnr = self.ht.compute_fpr_tpr_theoretical(null, alter, null_model, alter_model, threshold)
                else:
                    fpr, fnr = [], []
                    for i in range(len(alter_model)):
                        fpr_i, fnr_i = self.ht.compute_fpr_tpr_empirical(null, alter, null_model, alter_model[i],
                                                                         threshold)
                        fpr.append(fpr_i)
                        fnr.append(fnr_i)
                    fpr = np.stack(fpr, axis=0).mean(axis=0)
                    fnr = np.stack(fnr, axis=0).mean(axis=0)
            else:
                raise ValueError('Not valid ht mode')
        else:
            raise ValueError('Not valid ht mode')
        threshold = threshold.cpu().numpy()
        output = {'threshold': threshold, 'fpr': fpr, 'fnr': fnr}
        return output

    def update(self, output):
        self.result['threshold'].append(output['threshold'])
        self.result['fpr'].append(output['fpr'])
        self.result['fnr'].append(output['fnr'])
        return

    def state_dict(self):
        return self.result


def compute_fpr_tpr_empirical(y_true, y_score, threshold):
    # Expand dimensions to allow broadcasting
    y_true = y_true[:, None]  # Shape [N, 1]
    y_score = y_score[:, None]  # Shape [N, 1]

    # Generate predictions based on thresholds
    y_pred = (y_score >= threshold).float()  # Shape [N, T] where T is number of thresholds

    # Compute TP, TN, FP, FN
    TP = torch.sum((y_pred == 1) & (y_true == 1), dim=0).float()
    TN = torch.sum((y_pred == 0) & (y_true == 0), dim=0).float()
    FP = torch.sum((y_pred == 1) & (y_true == 0), dim=0).float()
    FN = torch.sum((y_pred == 0) & (y_true == 1), dim=0).float()

    # Compute FPR and FNR
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)

    return FPR.cpu().numpy(), FNR.cpu().numpy()


def bootstrap(data, num_samples):
    # Randomly sample indices with replacement
    indices = torch.randint(0, len(data), (num_samples,))
    # Get the samples
    samples = data[indices]
    return samples


def make_ht(ht_mode, num_samples_emp):
    return HypothesisTest(ht_mode, num_samples_emp)
