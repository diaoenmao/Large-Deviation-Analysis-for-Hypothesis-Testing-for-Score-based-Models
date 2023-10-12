import copy
import torch
import math
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
            if not isinstance(alter_model, list):
                data = torch.cat([null, alter], dim=0)
                target = torch.cat([torch.zeros(null.size(0)), torch.ones(alter.size(0))], dim=0).to(data.device)
                score = self.ht.score(data, null_model, alter_model)
                mask = torch.isfinite(score)
                score = score[mask]
                target = target[mask]
                _, _, threshold = roc_curve(target.cpu().numpy(), score.cpu().numpy())
                threshold = torch.tensor(threshold, device=data.device)
            else:
                null_split = null.split(null, cfg['num_samples_emp'], dim=0)
                alter_split = alter.split(alter, cfg['num_samples_emp'], dim=0)
                for i in range(len(alter_model)):
                    data_i = torch.cat([null_split[i], alter_split[i]], dim=0)
                    target_i = torch.cat([torch.zeros(null_split[i].size(0)), torch.ones(alter_split[i].size(0))], dim=0).to(data_i.device)
                    score_i = self.ht.score(data_i, null_model, alter_model[i])
                    mask = torch.isfinite(score_i)
                    score_i = score_i[mask]
                    target_i = target_i[mask]
                    _, _, threshold = roc_curve(target_i.cpu().numpy(), score_i.cpu().numpy())
                    threshold_i = torch.tensor(threshold, device=data_split_i.device)
                    print(len(threshold_i))
            exit()
        return threshold

    def test(self, input):
        null, alter, null_param, alter_param = input['null'], input['alter'], input['null_param'], input['alter_param']
        if self.ht_mode[0] in ['lrt', 'hst']:
            null_model = eval('model.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
            if self.num_samples_emp is not None:
                alter_mode = []
                alter_split = torch.split(alter, self.num_samples_emp, dim=0)
                for i in range(math.ceil(alter.size(0) // self.num_samples_emp)):
                    alter_model_i = eval('model.{}(copy.deepcopy(null_model.params)).to(cfg["device"])'.format(
                        cfg['model_name']))
                    print(alter_split[i].shape)
                    alter_model_i.fit(alter_split[i])
                    alter_mode.append(alter_model_i)
            else:
                alter_model = eval('model.{}(alter_param).to(cfg["device"])'.format(cfg['model_name']))
            threshold = self.make_threshold(null, alter, null_model, alter_model)
            if self.ht_mode[1] == 'e':
                with torch.no_grad():
                    data = torch.cat([null, alter], dim=0)
                    target = torch.cat([torch.zeros(null.size(0)), torch.ones(alter.size(0))], dim=0).to(data.device)
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

    return FPR.cpu().numpy(), FNR.cpu().numpy()


def make_ht(ht_mode, num_samples_emp):
    return HypothesisTest(ht_mode, num_samples_emp)
