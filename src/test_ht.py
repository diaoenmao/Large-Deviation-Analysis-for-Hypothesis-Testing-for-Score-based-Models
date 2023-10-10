import argparse
import os
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from dataset import make_dataset, make_data_loader, process_dataset, collate
from metric import make_metric, make_logger
from model import make_model
from module import save, load, to_device, process_control, resume, make_footprint, make_ht

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiment']))
    for i in range(cfg['num_experiment']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    process_control()
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    params = make_params(cfg['data_name'])
    print(params)
    exit()
    result_path = os.path.join('output', 'result')
    dataset = make_dataset(cfg['data_name'], params)
    dataset = process_dataset(dataset)
    null_model, alter_model = make_model(cfg['model_name'], params)
    data_loader = make_data_loader(dataset, cfg['model_name'])
    ht = make_ht(dataset, null_model, alter_model, cfg['ht_mode'])
    metric = make_metric({'test': ['AUROC']})
    logger = make_logger(os.path.join('output', 'runs', 'test_{}'.format(cfg['model_tag'])))
    test(data_loader['test'], model, ht, metric, logger)
    result = {'cfg': cfg, 'logger': logger, 'ht_state_dict': ht.state_dict()}
    save(result, os.path.join(result_path, cfg['model_tag']))
    return


def make_params(data_name):
    if data_name == 'MVN':
        mean = cfg['mvn']['mean']
        logvar = cfg['mvn']['logvar']
        ptb_mean, ptb_logvar = cfg['ptb'].split('-')
        ptb_mean, ptb_logvar = float(ptb_mean), float(ptb_logvar)
        params = {'num_trials': cfg['num_trials'], 'num_samples': cfg['num_samples'], 'mean': mean, 'logvar': logvar,
                  'ptb_mean': ptb_mean, 'ptb_logvar': ptb_logvar}
    elif data_name == 'RBM':
        W = cfg['rbm']['W']
        v = cfg['rbm']['v']
        h = cfg['rbm']['h']
        num_iters = cfg['rbm']['num_iters']
        ptb_W = float(cfg['ptb'])
        params = {'num_trials': cfg['num_trials'], 'num_samples': cfg['num_samples'], 'W': W, 'v': v, 'h': h,
                  'num_iters': num_iters, 'ptb_W': ptb_W}
    elif data_name == 'EXP':
        power = cfg['exp']['power']
        tau = cfg['exp']['tau']
        num_dims = cfg['exp']['num_dims']
        ptb_tau = float(cfg['ptb'])
        params = {'num_trials': cfg['num_trials'], 'num_samples': cfg['num_samples'], 'power': power, 'tau': tau,
                  'num_dims': num_dims, 'ptb_tau': ptb_tau}
    else:
        raise ValueError('Not valid data name')
    footprint = make_footprint(params)
    params = load(os.path.join('output', 'params', data_name, '{}_{}'.format(data_name, footprint)))
    return params


def test(data_loader, model, ht, metric, logger):
    start_time = time.time()
    for i, input in enumerate(data_loader):
        input = collate(input)
        print(input['null'].size())
        exit()
        input_size = input['null'].size(0)
        input = to_device(input, cfg['device'])
        output = ht.test(input, model)
        evaluation = metric.evaluate('test', 'batch', input, output)
        logger.append(evaluation, 'test', input_size)
        if i % np.ceil((len(data_loader) * cfg['log_interval'])) == 0:
            batch_time = (time.time() - start_time) / (i + 1)
            exp_finished_time = datetime.timedelta(seconds=round(batch_time * (len(data_loader) - i - 1)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Test Iter: {}/{}({:.0f}%)'.format(i + 1, len(data_loader), 100. * i / len(data_loader)),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'test')
            print(logger.write('test', metric.metric_name['test']))
        logger.save(True)
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Test Iter: {}/{}(100%)'.format(len(data_loader), len(data_loader))]}
    logger.append(info, 'test')
    print(logger.write('test', metric.metric_name['test']))
    return


if __name__ == "__main__":
    main()
