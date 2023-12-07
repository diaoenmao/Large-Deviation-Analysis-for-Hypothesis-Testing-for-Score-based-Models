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
        self.num_test_emp = 100
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
            null_score = make_score(null, null_model, alter_model, self.ht.score, 1)
            alter_score = make_score(alter, null_model, alter_model, self.ht.score, 1)
            score = torch.cat([null_score, alter_score], dim=0)
            min_value = torch.finfo(score.dtype).min
            max_value = torch.finfo(score.dtype).max
            score = torch.clamp(score, min=min_value, max=max_value)
            fpr, _, threshold = roc_curve(target.cpu().numpy(), score.cpu().numpy())
            threshold = np.interp(idx, fpr, threshold)
            threshold = torch.tensor(threshold, device=null.device, dtype=torch.float32)
        return threshold

    def test(self, input):
        null, alter, null_param, alter_param = input['null'], input['alter'], input['null_param'], input['alter_param']
        null_model = eval('model.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
        if cfg['data_name'] == 'KDDCUP99':
            null_model.fit(null)
        if self.num_samples_emp is not None:
            if self.num_samples_emp == -1:
                num_samples_emp = alter.size(0)
            else:
                num_samples_emp = self.num_samples_emp
            alter_model = []
            for i in range(self.num_test_emp):
                indices = torch.randint(0, len(alter), (num_samples_emp,))
                alter_i = alter[indices]
                alter_model_i = eval('model.{}(copy.deepcopy(null_model.params)).to(cfg["device"])'.format(
                    cfg['model_name']))
                alter_model_i.fit(alter_i)
                alter_model.append(alter_model_i)
        else:
            alter_model = [eval('model.{}(alter_param).to(cfg["device"])'.format(cfg['model_name']))]
        threshold = self.make_threshold(null, alter, null_model, alter_model)
        if self.ht_mode[0] in ['lrt', 'hst']:
            if self.ht_mode[1] == 'e':
                fpr, fnr = compute_empirical(null, alter, null_model, alter_model, threshold,
                                             self.ht.score)
            elif self.ht_mode[1] == 't':
                fpr, fnr = compute_theoretical(null, alter, null_model, alter_model,
                                               threshold, self.ht.score, self.optim_iter)
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
