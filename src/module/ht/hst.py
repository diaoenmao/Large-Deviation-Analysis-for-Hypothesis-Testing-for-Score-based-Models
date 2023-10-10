import copy
import numpy as np
import torch
import model
from scipy.stats import chi2
from config import cfg


class HST:
    def __init__(self):
        super().__init__()

    def compute_fpr_tpr_theoretical(self, null, alter, null_model, alter_model, threshold):
        pass

    def score(self, data, null_model, alter_model):
        """Calculate Hyvarinen Score Difference"""
        score = 2 * (null_model.hscore(data) - alter_model.hscore(data))
        score = score.reshape(-1)
        return score
