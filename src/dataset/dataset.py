import dataset
import numpy as np
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from config import cfg


def make_dataset(data_name, params=None, verbose=True):
    dataset_ = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = os.path.join('data', data_name)
    if data_name in ['MVN']:
        dataset_['test'] = dataset.MVN(root, **params)
    elif data_name in ['RBM']:
        dataset_['test'] = dataset.RBM(root, **params)
    elif data_name in ['KDDCUP99']:
        dataset_['test'] = dataset.KDDCUP99(root, **params)
    elif data_name in ['EXP']:
        dataset_['test'] = dataset.EXP(root, **params)
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset_


def input_collate(batch):
    return {key: [b[key] for b in batch] for key in batch[0]}


def make_data_collate(collate_mode):
    if collate_mode == 'dict':
        return input_collate
    elif collate_mode == 'default':
        return default_collate
    else:
        raise ValueError('Not valid collate mode')


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    cfg['num_steps'] = {}
    for k in dataset:
        batch_size_ = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        shuffle_ = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size_, shuffle=shuffle_,
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_worker'],
                                        collate_fn=make_data_collate(cfg['collate_mode']),
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=batch_size_, sampler=sampler[k],
                                        pin_memory=cfg['pin_memory'], num_workers=cfg['num_worker'],
                                        collate_fn=make_data_collate(cfg['collate_mode']),
                                        worker_init_fn=np.random.seed(cfg['seed']))
        cfg['num_steps'][k] = len(data_loader[k])
    return data_loader


def collate(input):
    for k in input:
        if k in ['null_param', 'alter_param']:
            input[k] = input[k][0]
        elif k in ['id', 'data', 'target']:
            input[k] = torch.stack(input[k], 0)
        else:
            input[k] = torch.cat(input[k], 0)
    return input

def process_dataset(dataset):
    processed_dataset = dataset
    cfg['data_size'] = {k: len(processed_dataset[k]) for k in processed_dataset}
    return processed_dataset


def split_dataset(dataset):
    dataset_ = []
    for i in range(cfg['target_size']):
        dataset_i = copy.deepcopy(dataset)
        mask = dataset_i['test'].target == i
        dataset_i['test'].id = dataset_i['test'].id[mask]
        dataset_i['test'].data = dataset_i['test'].data[mask]
        dataset_i['test'].target = dataset_i['test'].target[mask]
        dataset_i['test'].id = dataset_i['test'].id[:cfg['num_samples']]
        dataset_i['test'].data = dataset_i['test'].data[:cfg['num_samples']]
        dataset_i['test'].target = dataset_i['test'].target[:cfg['num_samples']]
        dataset_.append(dataset_i)
    return dataset_
