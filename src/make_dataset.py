import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from config import cfg, process_args
from dataset import make_dataset
from module import save, process_control, make_footprint

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

if __name__ == "__main__":
    cfg['seed'] = 0
    process_control()
    num_trials = cfg['num_trials']
    num_samples = cfg['num_samples']
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    # data_names = ['MVN', 'RBM', 'EXP']
    data_names = ['RBM']
    for m in range(len(data_names)):
        data_name = data_names[m]
        if data_name == 'MVN':
            mean = cfg['mvn']['mean']
            var = cfg['mvn']['var']
            ptb_mean = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9, 0.95,
                        1, 2]
            for i in range(len(ptb_mean)):
                ptb_mean_i = float(ptb_mean[i])
                ptb_logvar = float(0)
                params_i = {'num_trials': num_trials, 'num_samples': num_samples, 'mean': mean, 'var': var,
                            'ptb_mean': ptb_mean_i, 'ptb_logvar': ptb_logvar}
                dataset = make_dataset(data_name, params_i)
                footprint = make_footprint(params_i)
                save(params_i, os.path.join('output', 'params', data_name, '{}_{}'.format(data_name, footprint)))
            ptb_logvar = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9,
                          0.95, 1, 2]
            for i in range(len(ptb_logvar)):
                ptb_logvar_i = float(ptb_logvar[i])
                ptb_mean = float(0)
                params_i = {'num_trials': num_trials, 'num_samples': num_samples, 'mean': mean, 'var': var,
                            'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar_i}
                dataset = make_dataset(data_name, params_i)
                footprint = make_footprint(params_i)
                save(params_i, os.path.join('output', 'params', data_name, '{}_{}'.format(data_name, footprint)))
        elif data_name == 'RBM':
            W = cfg['rbm']['W']
            v = cfg['rbm']['v']
            h = cfg['rbm']['h']
            num_iters = cfg['rbm']['num_iters']
            ptb_W = [0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002, 0.0025,
                     0.003, 0.0035, 0.004, 0.0045, 0.005, 0.0075, 0.01]
            for i in range(len(ptb_W)):
                ptb_W_i = float(ptb_W[i])
                params_i = {'num_trials': num_trials, 'num_samples': num_samples, 'W': W, 'v': v, 'h': h,
                            'num_iters': num_iters, 'ptb_W': ptb_W_i}
                dataset = make_dataset(data_name, params_i)
                footprint = make_footprint(params_i)
                save(params_i, os.path.join('output', 'params', data_name, '{}_{}'.format(data_name, footprint)))
        elif data_name == 'EXP':
            power = cfg['exp']['power']
            tau = cfg['exp']['tau']
            num_dims = cfg['exp']['num_dims']
            ptb_tau = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
                       2.0]
            for i in range(len(ptb_tau)):
                ptb_tau_i = float(ptb_tau[i])
                params_i = {'num_trials': num_trials, 'num_samples': num_samples, 'power': power, 'tau': tau,
                            'num_dims': num_dims, 'ptb_tau': ptb_tau_i}
                dataset = make_dataset(data_name, params_i)
                footprint = make_footprint(params_i)
                save(params_i, os.path.join('output', 'params', data_name, '{}_{}'.format(data_name, footprint)))
        else:
            raise ValueError('Not valid data name')
