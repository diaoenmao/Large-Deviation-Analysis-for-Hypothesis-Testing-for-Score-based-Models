from config import cfg


def process_control():
    cfg['collate_mode'] = 'dict'
    data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                  'CIFAR100': [3, 32, 32]}
    cfg['linear'] = {}
    cfg['mlp'] = {'hidden_size': 128, 'scale_factor': 2, 'num_layers': 2, 'activation': 'relu'}
    cfg['cnn'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet9'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['resnet18'] = {'hidden_size': [64, 128, 256, 512]}
    cfg['wresnet28x2'] = {'depth': 28, 'widen_factor': 2, 'drop_rate': 0.0}
    cfg['wresnet28x8'] = {'depth': 28, 'widen_factor': 8, 'drop_rate': 0.0}
    cfg['data_name'] = cfg['control']['data_name']
    cfg['data_shape'] = data_shape[cfg['data_name']]
    cfg['model_name'] = cfg['control']['model_name']
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    cfg[model_name]['optimizer_name'] = 'SGD'
    cfg[model_name]['lr'] = 1e-1
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['weight_decay'] = 5e-4
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['num_epochs'] = 400
    cfg[model_name]['batch_size'] = {'train': 250, 'test': 250}
    cfg[model_name]['scheduler_name'] = 'CosineAnnealingLR'




    cfg['data_name'] = cfg['control']['data_name']
    if cfg['data_name'] in ['KDDCUP99']:
        cfg['model_name'] = 'rbm'
    else:
        cfg['model_name'] = cfg['data_name'].lower()
    if 'test_mode' in cfg['control']:
        cfg['test_mode'] = cfg['control']['test_mode']
    if 'ptb' in cfg['control']:
        cfg['ptb'] = cfg['control']['ptb']
    if 'alter_num_samples' in cfg['control']:
        cfg['alter_num_samples'] = int(cfg['control']['alter_num_samples'])
    if 'alter_noise' in cfg['control']:
        cfg['alter_noise'] = float(cfg['control']['alter_noise'])
    # cfg['num_trials'] = 100
    # cfg['num_samples'] = 10000
    cfg['num_trials'] = 10
    # cfg['num_trials'] = 1
    cfg['num_samples'] = 10000
    cfg['gof'] = {}
    cfg['gof']['batch_size'] = {'test': 1}
    cfg['gof']['shuffle'] = {'test': False}
    cfg['gof']['drop_last'] = {'test': False}
    if cfg['data_name'] in ['KDDCUP99']:
        data_shape = {'MNIST': [1, 28, 28], 'FashionMNIST': [1, 28, 28], 'SVHN': [3, 32, 32], 'CIFAR10': [3, 32, 32],
                      'CIFAR100': [3, 32, 32], 'KDDCUP99': [39]}
        cfg['data_shape'] = data_shape[cfg['data_name']]
        null_label = {'KDDCUP99': 4}
        cfg['null_label'] = null_label[cfg['data_name']]
        dim_v = cfg['data_shape'][0]
        # dim_v = 70
        # dim_h = 20
        dim_h = 50
        # dim_h = 60
        generator = torch.Generator()
        generator.manual_seed(cfg['seed'])
        W = torch.randn(dim_v, dim_h, generator=generator)
        v = torch.randn(dim_v, generator=generator)
        h = torch.randn(dim_h, generator=generator)
        cfg['rbm'] = {'W': W, 'v': v, 'h': h, 'num_iters': int(1)}
    else:
        d = 2
        if d == 1:
            cfg['mvn'] = {'mean': torch.tensor([0.]), 'logvar': torch.tensor([1.])}
            cfg['gmm'] = {'mean': torch.tensor([[0.], [2.], [4.]]),
                          'logvar': torch.tensor([[0.], [0.2], [0.4]]),
                          'logweight': torch.log(torch.tensor([0.2, 0.6, 0.2])),
                          'num_components': 3}
            dim_v = 1
            dim_h = 2
            generator = torch.Generator()
            generator.manual_seed(cfg['seed'])
            W = torch.randn(dim_v, dim_h, generator=generator)
            v = torch.randn(dim_v, generator=generator)
            h = torch.randn(dim_h, generator=generator)
            cfg['rbm'] = {'W': W, 'v': v, 'h': h, 'num_iters': int(100)}
            cfg['exp'] = {'power': torch.tensor([4.]), 'tau': torch.tensor([1.]), 'num_dims': torch.tensor([1])}
        else:
            cfg['mvn'] = {'mean': torch.tensor([0., 0.]), 'logvar': torch.tensor([[0., -0.3], [-0.3, 0.]])}
            # cfg['gmm'] = {'mean': torch.tensor([[0., 0.], [4., 0.], [0., 4.]]),
            #               'logvar': torch.tensor([[[0., -0.3], [-0.3, 0.]],
            #                                       [[0.2, -0.3], [-0.3, 0.2]],
            #                                       [[0.4, -0.3], [-0.3, 0.4]]]),
            #               'logweight': torch.log(torch.tensor([0.2, 0.6, 0.2])),
            #               'num_components': 3}
            cfg['gmm'] = {'mean': torch.tensor([[0.], [2.], [4.]]),
                          'logvar': torch.tensor([[0.], [0.2], [0.4]]),
                          'logweight': torch.log(torch.tensor([0.2, 0.6, 0.2])),
                          'num_components': 3}
            # dim_v = 30
            dim_v = 50
            # dim_v = 70
            # dim_h = 20
            dim_h = 40
            # dim_h = 60
            generator = torch.Generator()
            generator.manual_seed(cfg['seed'])
            W = torch.randn(dim_v, dim_h, generator=generator)
            v = torch.randn(dim_v, generator=generator)
            h = torch.randn(dim_h, generator=generator)
            cfg['rbm'] = {'W': W, 'v': v, 'h': h, 'num_iters': int(1000)}
            cfg['exp'] = {'power': torch.tensor([4.]), 'tau': torch.tensor([1.]), 'num_dims': torch.tensor([3])}
    cfg['hst'] = {}
    cfg['hst']['optimizer_name'] = 'Adam'
    cfg['hst']['lr'] = 1e-3
    cfg['hst']['betas'] = (0.9, 0.999)
    cfg['hst']['momentum'] = 0.9
    cfg['hst']['nesterov'] = True
    cfg['hst']['weight_decay'] = 0
    cfg['hst']['num_iters'] = 20
    cfg['hst']['drop_last'] = {'train': False, 'test': False}
    cfg['num_bootstrap'] = 1000
    cfg['alpha'] = 0.05
    cfg['ood'] = {}
    cfg['ood']['optimizer_name'] = 'Adam'
    cfg['ood']['scheduler_name'] = 'None'
    cfg['ood']['lr'] = 1e-3
    cfg['ood']['betas'] = (0.9, 0.999)
    cfg['ood']['momentum'] = 0.9
    cfg['ood']['nesterov'] = True
    cfg['ood']['weight_decay'] = 0
    cfg['ood']['num_epochs'] = 20
    cfg['ood']['batch_size'] = {'train': 250, 'test': 500}
    cfg['ood']['shuffle'] = {'train': True, 'test': False}
    cfg['ood']['drop_last'] = {'train': False, 'test': False}
    return
