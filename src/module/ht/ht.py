import copy
import torch
import model
from config import cfg
from .lrt import LRT
from .hst import HST


class HypothesisTest:
    def __init__(self, ht_mode):
        self.ht_mode = ht_mode.split('-')
        self.ht = self.make_ht()
        self.result = {'threshold': [], 'fpr': [], 'fnr': []}

    def make_ht(self):
        if self.ht_mode[0] in ['lrt']:
            ht = LRT()
        elif self.ht_mode[0] in ['hst']:
            ht = HST()
        else:
            raise ValueError('Not valid ht mode')
        return ht

    def make_threshold(self, null_samples, alter_samples, null_model, alter_model):
        with torch.no_grad():
            data = torch.cat([null_samples, alter_samples], dim=0)
            target = torch.cat([torch.zeros(null_samples.size(0)), torch.ones(alter_samples.size(0))], dim=0)
            score = self.ht.score(data, null_model, alter_model)
            _, _, threshold = roc_curve(target.numpy(), score.numpy())
        return threshold

    def test(self, input):
        num_samples_emp = cfg['num_samples_emp']
        null, alter, null_param, alter_param = input['null'], input['alter'], input['null_param'], input['alter_param']
        if self.ht_mode in ['lrt-t', 'lrt-e', 'hst-t', 'hst-e']:
            null_model = eval('models.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
            if self.ht_mode[1] == 'e':
                null_model_emp = eval('model.{}(copy.deepcopy(null_model.params)).to(cfg["device"])'.format(
                    cfg['model_name']))
                alter_model = eval('model.{}(copy.deepcopy(null_model.params)).to(cfg["device"])'.format(
                    cfg['model_name']))
                null_model_emp.fit(null[:num_samples_emp])
                alter_model.fit(alter[:num_samples_emp])
                null_model = null_model_emp
            else:
                alter_model = eval('models.{}(alter_param).to(cfg["device"])'.format(cfg['model_name']))
            threshold = self.make_threshold(null, alter, null_model, alter_model)
            if self.ht_mode[1] == 'e':
                with torch.no_grad():
                    data = torch.cat([null_samples, alter_samples], dim=0)
                    target = torch.cat([torch.zeros(null_samples.size(0)), torch.ones(alter_samples.size(0))], dim=0)
                    score = self.ht.score(data, null_model, alter_model)
                    fpr, fnr = compute_fpr_tpr_empirical(target, score, threshold)
            elif self.ht_mode[1] == 't':
                fpr, fnr = self.ht.compute_fpr_tpr_theoretical(null, alter, null_model, alter_model, threshold)
            else:
                raise ValueError('Not valid ht mode')
        else:
            raise ValueError('Not valid ht mode')
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

    return FPR.numpy(), FNR.numpy()


def make_ht(ht_mode):
    return HypothesisTest(ht_mode)
