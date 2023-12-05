import copy
import torch
import numpy as np
import model
from config import cfg
from sklearn.metrics import roc_curve
from .lrt import LRT
from .hst import HST
from .utils import make_score, compute_empirical, compute_theoretical


class HypothesisTest:
    def __init__(self, ht_mode, num_samples_emp):
        self.ht_mode = ht_mode.split('-')
        self.num_samples_emp = num_samples_emp
        self.ht = self.make_ht()
        self.result = {'threshold': [], 'fpr': [], 'fnr': []}
        self.num_threshold = 3000
        self.num_test_emp = 1
        self.optim_iter = 50

    def make_ht(self):
        if self.ht_mode[0] in ['lrt']:
            ht = LRT()
        elif self.ht_mode[0] in ['hst']:
            ht = HST()
        else:
            raise ValueError('Not valid ht mode')
        return ht

    def make_threshold(self, null, alter, null_model, alter_model):
        with torch.no_grad():
            target = torch.cat([torch.zeros(null.size(0)), torch.ones(alter.size(0))], dim=0).to(null.device)
            idx = np.linspace(0, 1, self.num_threshold)
            threshold = []
            for i in range(len(alter_model)):
                null_score_i = make_score(null, null_model, alter_model[i], self.ht.score, 1)
                alter_score_i = make_score(alter, null_model, alter_model[i], self.ht.score, 1)
                score_i = torch.cat([null_score_i, alter_score_i], dim=0)
                min_value = torch.finfo(score_i.dtype).min
                max_value = torch.finfo(score_i.dtype).max
                score_i = torch.clamp(score_i, min=min_value, max=max_value)
                fpr_i, _, threshold_i = roc_curve(target.cpu().numpy(), score_i.cpu().numpy())
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
            alter_model = [eval('model.{}(alter_param).to(cfg["device"])'.format(cfg['model_name']))]
        threshold = self.make_threshold(null, alter, null_model, alter_model)
        if self.ht_mode[0] in ['lrt', 'hst']:
            if self.ht_mode[1] == 'e':
                with torch.no_grad():
                    target = torch.cat([torch.zeros(null.size(0)), torch.ones(alter.size(0))], dim=0).to(null.device)
                    fpr, fnr = [], []
                    for i in range(len(alter_model)):
                        fpr_i, fnr_i = compute_empirical(null, alter, null_model, alter_model[i], threshold,
                                                                 self.ht.score, target)
                        fpr.append(fpr_i)
                        fnr.append(fnr_i)
                    fpr = np.stack(fpr, axis=0).mean(axis=0)
                    fnr = np.stack(fnr, axis=0).mean(axis=0)
            elif self.ht_mode[1] == 't':
                fpr, fnr = [], []
                for i in range(len(alter_model)):
                    fpr_i, fnr_i = compute_theoretical(null, alter, null_model, alter_model[i],
                                                               threshold, self.ht.score, self.optim_iter)
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


def make_ht(ht_mode, num_samples_emp):
    return HypothesisTest(ht_mode, num_samples_emp)
