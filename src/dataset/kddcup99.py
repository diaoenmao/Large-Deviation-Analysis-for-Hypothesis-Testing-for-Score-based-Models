import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from module import check_exists, makedir_exist_ok, save, load, make_footprint


class KDDCUP99(Dataset):
    data_name = 'KDDCUP99'

    def __init__(self, root, **params):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.ptb = params['ptb']
        self.footprint = make_footprint(params)
        split_name = '{}_{}'.format(self.data_name, self.footprint)
        if not check_exists(os.path.join(self.processed_folder)):
            self.process()
        self.null, self.alter, self.meta = load(os.path.join(os.path.join(self.processed_folder, split_name)))

    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), torch.tensor(self.data[index]), torch.tensor(
            self.target[index])
        input = {'id': id, 'data': data, 'target': target}
        return input

    def __len__(self):
        return len(self.data)

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
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nnSplit: {}'.format(self.data_name, self.__len__(), self.root,
                                                                      self.split)
        return fmt_str

    def make_data(self):
        # https://github.com/timeamagyar/kdd-cup-99-python/blob/master/kdd%20preprocessing.ipynb
        from sklearn.preprocessing import LabelEncoder, Normalizer
        from sklearn.datasets import fetch_kddcup99
        dataset = fetch_kddcup99(as_frame=True)
        dataframe, target = dataset['data'], dataset['target']
        dataframe['label'] = target
        dataframe = dataframe.drop('num_outbound_cmds', axis=1)
        dataframe = dataframe.drop('is_host_login', axis=1)
        dataframe['protocol_type'] = dataframe['protocol_type'].astype('category')
        dataframe['service'] = dataframe['service'].astype('category')
        dataframe['flag'] = dataframe['flag'].astype('category')
        cat_columns = dataframe.select_dtypes(['category']).columns
        dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)
        dataframe = dataframe.drop_duplicates(subset=None, keep='first')

        minimal_samples = 100
        value_counts = dataframe['label'].value_counts()
        other_index = value_counts.index[value_counts.values < minimal_samples]
        for i in range(len(other_index)):
            dataframe.loc[dataframe['label'] == other_index[i], 'label'] = b'unknown'

        le = LabelEncoder()
        dataframe['label'] = le.fit_transform(dataframe['label'])
        data = dataframe.values[:, :-1].astype(np.float32)
        target = dataframe.values[:, -1].astype(np.int64)
        norm = Normalizer()
        data = norm.fit_transform(data)
        classes = le.classes_
        null_label = classes.tolist().index(b'normal.')
        normal_data = data[target == null_label]
        ptb_class = bytes(self.ptb, 'utf-8')
        for c in classes:
            if ptb_class in c:
                abnormal_label = classes.tolist().index(c)
                abnormal_data = data[target == abnormal_label]
                break
        null = normal_data
        alter = abnormal_data
        meta = {'class': ptb_class}
        return null, alter, meta

