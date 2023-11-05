import torch
from config import cfg


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


def make_score(data, null_model, alter_model, score_fn, batch_size=100):
    score = []
    for i in range(data.size(0) // batch_size):
        num_samples = cfg['num_samples_test'] * batch_size
        indices = torch.randint(0, len(data), (num_samples,))
        data_i = data[indices]
        score_i = score_fn(data_i, null_model, alter_model)
        score_i = score_i.view(cfg['num_samples_test'], -1).mean(dim=0)
        score.append(score_i)
    score = torch.cat(score)
    return score
