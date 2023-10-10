import copy
import numpy as np
import torch
import model
from scipy.stats import chi2
from config import cfg


class LRT:
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def test(self, null_samples, alter_samples, null_model, alter_model=None):
        num_tests = alter_samples.size(0)
        num_samples_alter = alter_samples.size(1)
        statistic = []
        pvalue = []
        for i in range(num_tests):
            if alter_model is None:
                null_model_emp = eval('model.{}(copy.deepcopy(null_model.params)).to(cfg["device"])'.format(
                    cfg['model_name']))
                null_model_emp.fit(null_samples[i])
                alter_model = eval('model.{}(copy.deepcopy(null_model.params)).to(cfg["device"])'.format(
                    cfg['model_name']))
                alter_model.fit(alter_samples[i])
            else:
                null_model_emp = alter_model
            with torch.no_grad():
                bootstrap_null_samples = self.m_out_n_bootstrap(null_samples, num_samples_alter, null_model,
                                                                null_model_emp)
                statistic_i, pvalue_i = self.density_test(alter_samples[i], bootstrap_null_samples, null_model,
                                                          alter_model, self.bootstrap_approx)
                statistic.append(statistic_i)
                pvalue.append(pvalue_i)
        return statistic, pvalue

    def lrt(self, samples, null_pdf, alter_pdf):
        """Calculate Likelihood Ratio"""
        LRT_items = 2 * (torch.log(alter_pdf(samples)) - torch.log(null_pdf(samples)))
        LRT_items = LRT_items.reshape(-1)
        test_statistic = torch.sum(LRT_items, -1)
        return LRT_items, test_statistic

    def density_test(self, alter_samples, bootstrap_null_samples, null_model, alter_model, bootstrap_approx):
        _, test_statistic = self.lrt(alter_samples, null_model.pdf, alter_model.pdf)
        test_statistic = test_statistic.item()
        if bootstrap_approx:
            pvalue = torch.mean((bootstrap_null_samples >= test_statistic).float()).item()
        else:
            df = 1
            pvalue = 1 - chi2(df).cdf(test_statistic)  # since Λ follows χ2
        return test_statistic, pvalue
