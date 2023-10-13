import os
import torch
import hashlib
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load, make_footprint


class MVN(Dataset):
    data_name = 'MVN'

    def __init__(self, root, **params):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.num_trials = params['num_trials']
        self.num_samples = params['num_samples']
        self.mean = params['mean']
        self.var = params['var']
        self.ptb_mean = params['ptb_mean']
        self.ptb_logvar = params['ptb_logvar']
        self.footprint = make_footprint(params)
        split_name = '{}_{}'.format(self.data_name, self.footprint)
        if not check_exists(os.path.join(self.processed_folder, split_name)):
            print('Not exists {}, create from scratch with {}.'.format(split_name, params))
            self.process()
        self.null, self.alter, self.meta = load(os.path.join(os.path.join(self.processed_folder, split_name)))
    def __getitem__(self, index):
        null, alter = torch.tensor(self.null[index]), torch.tensor(self.alter[index])
        null_param = {'mean': self.mean, 'var': self.var}
        alter_param = {'mean': torch.tensor(self.meta['mean'][index]),
                       'var': torch.tensor(self.meta['var'][index])}
        input = {'null': null, 'alter': alter, 'null_param': null_param, 'alter_param': alter_param}
        return input

    def __len__(self):
        return self.num_trials

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        dataset = self.make_data()
        save(dataset, os.path.join(self.processed_folder, '{}_{}'.format(self.data_name, self.footprint)))
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nFootprint: {}'.format(self.data_name, self.__len__(), self.root,
                                                                         self.footprint)
        return fmt_str

    def make_data(self):
        total_samples = self.num_trials * self.num_samples
        d = self.mean.size(-1)
        if d == 1:
            null_mvn = torch.distributions.normal.Normal(self.mean, self.var.sqrt())
        else:
            null_mvn = torch.distributions.multivariate_normal.MultivariateNormal(self.mean, self.var)
        null = null_mvn.sample((total_samples,))
        null = null.view(self.num_trials, self.num_samples, -1)
        ptb_mean = self.ptb_mean * torch.randn((self.num_trials, *self.mean.size()))
        alter_mean = self.mean + ptb_mean
        if d == 1:
            ptb_logvar = self.ptb_logvar * torch.randn((self.num_trials, *self.var.size()))
            alter_var = (self.var.log() + ptb_logvar).exp()
        else:
            alter_var = []
            for i in range(self.num_trials):
                pd_flag = False
                while not pd_flag:
                    ptb_logvar_i = self.ptb_logvar * torch.randn((d,))
                    alter_var_i_diag = (torch.diag(self.var).log() + ptb_logvar_i).exp()
                    alter_var_i = self.var.clone()

                    rows = torch.arange(alter_var_i.size(0))
                    cols = torch.arange(alter_var_i.size(1))
                    alter_var_i[rows, cols] = alter_var_i_diag
                    if (torch.linalg.eigvalsh(alter_var_i) > 0).all():
                        pd_flag = True
                        alter_var.append(alter_var_i)
            alter_var = torch.stack(alter_var, dim=0)
        if d == 1:
            alter_mvn = torch.distributions.normal.Normal(alter_mean, alter_var.sqrt())
        else:
            alter_mvn = torch.distributions.multivariate_normal.MultivariateNormal(alter_mean, alter_var)
        alter = alter_mvn.sample((self.num_samples,))
        alter = alter.permute(1, 0, 2)
        null, alter = null.numpy(), alter.numpy()
        meta = {'mean': alter_mean.numpy(), 'var': alter_var.numpy()}
        return null, alter, meta
