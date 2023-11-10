import torch


class HST:
    def __init__(self):
        super().__init__()

    def score(self, data, null_model, alter_model):
        """Calculate Hyvarinen Score Difference"""
        score = null_model.hscore(data) - alter_model.hscore(data)
        score = score.reshape(-1)
        return score
