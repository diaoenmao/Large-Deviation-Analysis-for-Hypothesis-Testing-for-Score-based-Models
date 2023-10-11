import copy
import numpy as np
import torch
import model
from config import cfg


class LRT:
    def __init__(self):
        super().__init__()

    def compute_fpr_tpr_theoretical(self, null, alter, null_model, alter_model, threshold):
        pass

    def score(self, data, null_model, alter_model):
        """Calculate Likelihood Ratio"""
        score = 2 * (torch.log(alter_model.pdf(data)) - torch.log(null_model.pdf(data)))
        score = score.reshape(-1)
        return score


