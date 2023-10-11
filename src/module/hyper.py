import torch
from config import cfg


def process_control():
    cfg['collate_mode'] = 'dict'
    cfg['data_name'] = cfg['control']['data_name']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['ht_mode'] = cfg['control']['ht_mode']
    cfg['ptb'] = cfg['control']['ptb']
    if 'num_samples_emp' in cfg['control']:
        cfg['num_samples_emp'] = int(cfg['control']['num_samples_emp'])
    else:
        cfg['num_samples_emp'] = None
    cfg['num_trials'] = 10
    cfg['num_samples'] = 10000
    if cfg['data_name'] in ['KDDCUP99']:
        data_shape = {'KDDCUP99': [39]}
        cfg['data_shape'] = data_shape[cfg['data_name']]
        null_label = {'KDDCUP99': 4}
        cfg['null_label'] = null_label[cfg['data_name']]
        dim_v = cfg['data_shape'][0]
        dim_h = 50
        generator = torch.Generator()
        generator.manual_seed(cfg['seed'])
        W = torch.randn(dim_v, dim_h, generator=generator)
        v = torch.randn(dim_v, generator=generator)
        h = torch.randn(dim_h, generator=generator)
        cfg['rbm'] = {'W': W, 'v': v, 'h': h, 'num_iters': int(1)}
    else:
        cfg['mvn'] = {'mean': torch.tensor([0., 0.]), 'logvar': torch.tensor([[0., -0.3], [-0.3, 0.]])}
        dim_v = 50
        dim_h = 40
        generator = torch.Generator()
        generator.manual_seed(cfg['seed'])
        W = torch.randn(dim_v, dim_h, generator=generator)
        v = torch.randn(dim_v, generator=generator)
        h = torch.randn(dim_h, generator=generator)
    cfg['rbm'] = {'W': W, 'v': v, 'h': h, 'num_iters': int(1000)}
    cfg['exp'] = {'power': torch.tensor([4.]), 'tau': torch.tensor([1.]), 'num_dims': torch.tensor([3])}
    model_name = cfg['model_name']
    cfg[model_name]['batch_size'] = {'test': 1}
    cfg[model_name]['shuffle'] = {'test': False}
    cfg[model_name]['optimizer_name'] = 'Adam'
    cfg[model_name]['lr'] = 1e-3
    cfg[model_name]['betas'] = (0.9, 0.999)
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['weight_decay'] = 0
    cfg[model_name]['num_iters'] = 20
    return
