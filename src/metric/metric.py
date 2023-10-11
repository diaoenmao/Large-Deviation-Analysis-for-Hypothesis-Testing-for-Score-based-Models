import torch
import torch.nn.functional as F
from collections import defaultdict
from config import cfg
from module import recur
from sklearn.metrics import auc


def make_metric(metric_name):
    if cfg['data_name'] in ['KDDCUP99']:
        pivot = float('inf')
        pivot_direction = 'down'
        pivot_name = 'Loss'
    else:
        pivot = None
        pivot_name = None
        pivot_direction = None
    metric = Metric(metric_name, pivot, pivot_direction, pivot_name)
    return metric


def AUROC(fpr, fnr):
    auroc = auc(fpr, 1 - fnr)
    return auroc


class Metric:
    def __init__(self, metric_name, pivot, pivot_direction, pivot_name):
        self.pivot, self.pivot_name, self.pivot_direction = pivot, pivot_name, pivot_direction
        self.metric_name = metric_name
        self.metric = self.make_metric(metric_name)

    def make_metric(self, metric_name):
        metric = defaultdict(dict)
        for split in metric_name:
            for m in metric_name[split]:
                if m == 'Loss':
                    metric[split][m] = {'mode': 'batch', 'metric': (lambda input, output: output['loss'].item())}
                elif m == 'AUROC':
                    metric[split][m] = {'mode': 'batch',
                                        'metric': (lambda input, output: recur(AUROC, output['fpr'], output['fnr']))}
                else:
                    raise ValueError('Not valid metric name')
        return metric

    def add(self, split, input, output):
        for metric_name in self.metric_name[split]:
            if self.metric[split][metric_name]['mode'] == 'full':
                self.metric[split][metric_name]['metric'].add(input, output)
        return

    def evaluate(self, split, mode, input=None, output=None, metric_name=None):
        metric_name = self.metric_name if metric_name is None else metric_name
        evaluation = {}
        for metric_name_ in metric_name[split]:
            if self.metric[split][metric_name_]['mode'] == mode:
                evaluation[metric_name_] = self.metric[split][metric_name_]['metric'](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return

    def load_state_dict(self, state_dict):
        self.pivot = state_dict['pivot']
        self.pivot_name = state_dict['pivot_name']
        self.pivot_direction = state_dict['pivot_direction']
        return

    def state_dict(self):
        return {'pivot': self.pivot, 'pivot_name': self.pivot_name, 'pivot_direction': self.pivot_direction}
