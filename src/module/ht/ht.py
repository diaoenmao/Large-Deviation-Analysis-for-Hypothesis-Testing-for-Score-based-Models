import torch
import model
from config import cfg
from .lrt import LRT
from .hst import HST


class HypothesisTest:
    def __init__(self, null_data, ht_mode):
        self.ht_mode = ht_mode
        self.ht = self.make_ht(null_data)
        self.threshold = []

    def make_ht(self):
        ht_mode_list = self.ht_mode.split('-')
        if self.ht_mode in ['lrt-t', 'lrt-e']:
            ht = LRT(model, ht_mode_list[1])
        elif self.ht_mode in ['hst-t', 'hst-e']:
            ht = HST(model, ht_mode_list[1])
        else:
            raise ValueError('Not valid ht mode')
        return ht

    def test(self, input):
        alter_noise = cfg['alter_noise']
        alter_num_samples = cfg['alter_num_samples']
        null, alter, null_param, alter_param = input['null'], input['alter'], input['null_param'], input['alter_param']
        alter = alter + alter_noise * torch.randn(alter.size(), device=alter.device)
        null_samples = torch.split(null, alter_num_samples, dim=0)
        alter_samples = torch.split(alter, alter_num_samples, dim=0)
        if len(null_samples) % alter_num_samples != 0:
            null_samples = null_samples[:-1]
        null_samples = torch.stack(null_samples, dim=0)
        if len(alter_samples) % alter_num_samples != 0:
            alter_samples = alter_samples[:-1]
        alter_samples = torch.stack(alter_samples, dim=0)
        if self.test_mode in ['cvm', 'ks']:
            alter_samples = alter_samples.cpu().numpy()
            null_model = eval('model.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
            statistic_t1, pvalue_t1 = self.gof.test(null_samples, null_model)
            statistic_t2, pvalue_t2 = self.gof.test(alter_samples, null_model)
        elif self.test_mode in ['ksd-u', 'ksd-v']:
            null_samples = null_samples
            alter_samples = alter_samples
            null_model = eval('model.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
            null_samples_permute = null_samples[torch.randperm(len(null_samples))]
            statistic_t1, pvalue_t1 = self.gof.test(null_samples, null_samples_permute, null_model)
            statistic_t2, pvalue_t2 = self.gof.test(null_samples, alter_samples, null_model)
        elif self.test_mode in ['mmd']:
            null_samples = null_samples
            alter_samples = alter_samples
            null_samples_permute = null_samples[torch.randperm(len(null_samples))]
            statistic_t1, pvalue_t1 = self.gof.test(null_samples, null_samples_permute)
            statistic_t2, pvalue_t2 = self.gof.test(null_samples, alter_samples)
        elif self.test_mode in ['lrt-chi2-g', 'lrt-b-g']:
            null_samples = null_samples
            alter_samples = alter_samples
            null_samples_permute = null_samples[torch.randperm(len(null_samples))]
            null_model = eval('model.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
            alter_model = eval('model.{}(alter_param).to(cfg["device"])'.format(cfg['model_name']))
            statistic_t1, pvalue_t1 = self.gof.test(null_samples, null_samples_permute, null_model, alter_model)
            statistic_t2, pvalue_t2 = self.gof.test(null_samples, alter_samples, null_model, alter_model)
        elif self.test_mode in ['lrt-chi2-e', 'lrt-b-e']:
            null_samples = null_samples
            alter_samples = alter_samples
            null_samples_permute = null_samples[torch.randperm(len(null_samples))]
            null_model = eval('model.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
            statistic_t1, pvalue_t1 = self.gof.test(null_samples, null_samples_permute, null_model)
            statistic_t2, pvalue_t2 = self.gof.test(null_samples, alter_samples, null_model)
        elif self.test_mode in ['hst-chi2-g', 'hst-b-g']:
            null_samples = null_samples
            alter_samples = alter_samples
            null_samples_permute = null_samples[torch.randperm(len(null_samples))]
            null_model = eval('model.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
            alter_model = eval('model.{}(alter_param).to(cfg["device"])'.format(cfg['model_name']))
            statistic_t1, pvalue_t1 = self.gof.test(null_samples, null_samples_permute, null_model, alter_model)
            statistic_t2, pvalue_t2 = self.gof.test(null_samples, alter_samples, null_model, alter_model)
        elif self.test_mode in ['hst-chi2-e', 'hst-b-e']:
            null_samples = null_samples
            alter_samples = alter_samples
            null_samples_permute = null_samples[torch.randperm(len(null_samples))]
            null_model = eval('model.{}(null_param).to(cfg["device"])'.format(cfg['model_name']))
            statistic_t1, pvalue_t1 = self.gof.test(null_samples, null_samples_permute, null_model)
            statistic_t2, pvalue_t2 = self.gof.test(null_samples, alter_samples, null_model)
        else:
            raise ValueError('Not valid test mode')
        output = {'statistic_t1': statistic_t1, 'pvalue_t1': pvalue_t1, 'statistic_t2': statistic_t2,
                  'pvalue_t2': pvalue_t2}
        return output


def make_ht(dataset, model, ht_mode):
    null_data = torch.tensor(dataset['test'].null)
    return HypothesisTest(null_data, model, ht_mode)