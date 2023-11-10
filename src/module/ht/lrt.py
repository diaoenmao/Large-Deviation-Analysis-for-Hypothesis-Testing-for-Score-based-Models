import torch


class LRT:
    def __init__(self):
        super().__init__()

    def score(self, data, null_model, alter_model):
        """Calculate Likelihood Ratio"""
        score = torch.log(alter_model.pdf(data)) - torch.log(null_model.pdf(data))
        score = score.reshape(-1)
        return score
