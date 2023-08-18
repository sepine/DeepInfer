# !/usr/bin/env python
# encoding: utf-8


import re
import os
from tqdm import tqdm
import numpy as np
import pickle
import logging
import torch
import random
from utils import gene_bit2dec, is_number, get_classify_types
from torch_geometric.data import Data, Dataset

logger = logging.getLogger(__name__)


def load_datasets_and_vocabs(args, type2idx, idx2type):

    dataset_path = os.path.join(args.base_dataset_path, args.language, args.module)

    # ##############################  Cache Data ########################################
    cached_path = args.cached_dataset_path

    cached_path_prefix = os.path.join(cached_path, args.language, args.module)
    if not os.path.exists(cached_path_prefix):
        os.makedirs(cached_path_prefix)

    cached_data_path = os.path.join(cached_path, args.language, args.module, 'cached_'
                                   + args.model + '_pos_' + str(args.pos) + '_data.pkl')

    if os.path.exists(cached_data_path):
        with open(cached_data_path, 'rb') as fr:
            train_dataset, test_dataset = pickle.load(fr)
        # logger.info('Load data from caches. The total number of train dataset is %d' % len(train_dataset))
        # logger.info('Load data from caches. The total number of test dataset is %d' % len(test_dataset))
        return train_dataset, test_dataset
    # ##############################  Cache Data ########################################

    with open(os.path.join(dataset_path, 'return_data.pkl'), 'rb') as fr:
        return_dataset = pickle.load(fr)

    # if args.language == 'compiler':
    if args.language.startswith('compiler'):
        train, test = split_train_test_compiler(return_dataset, args)
    else:
        total_cnt = len(return_dataset)
        split_cnt = (total_cnt // 5) * 4
        train = return_dataset[:split_cnt]
        random.shuffle(train)

        test = return_dataset[split_cnt:]

    train_dataset = TypeDataset(args.base_dataset_path, train, type2idx, idx2type, args)
    test_dataset = TypeDataset(args.base_dataset_path, test, type2idx, idx2type, args)

    # logger.info('Build Train finished! The total number is %d' % len(train_dataset))
    # logger.info('Build Test finished! The total number is %d' % len(test_dataset))

    # ##############################  Cache Data ########################################

    with open(cached_data_path, 'wb') as fw:
        pickle.dump((train_dataset, test_dataset), fw)

    # logger.info('Save data to caches successfully. The total number of train dataset is %d' % len(train_dataset))
    # logger.info('Save data to caches successfully. The total number of test dataset is %d' % len(test_dataset))

    # ##############################  Cache Data ########################################

    return train_dataset, test_dataset


def split_train_test_compiler(dataset, args):
    trains = []
    tests = []

    version1 = ['0.1.1', '0.1.2', '0.1.3', '0.1.4', '0.1.5', '0.1.6', '0.1.7',
                '0.2.0', '0.2.1', '0.2.2',
                '0.3.0', '0.3.1', '0.3.2', '0.3.3', '0.3.4', '0.3.5', '0.3.6',
                '0.4.0', '0.4.1', '0.4.2', '0.4.3', '0.4.4', '0.4.5', '0.4.6', '0.4.7', '0.4.8', '0.4.9', '0.4.10',
                '0.4.11', '0.4.12', '0.4.13', '0.4.14', '0.4.15', '0.4.16', '0.4.17', '0.4.18', '0.4.19', '0.4.20',
                '0.4.21', '0.4.22', '0.4.23', '0.4.24', '0.4.25', '0.4.26']

    version2 = ['0.5.0', '0.5.1', '0.5.2', '0.5.3', '0.5.4', '0.5.5', '0.5.6', '0.5.7', '0.5.8',
                '0.5.9', '0.5.10', '0.5.11', '0.5.12', '0.5.13', '0.5.14', '0.5.15', '0.5.16', '0.5.17']

    version3 = ['0.6.0', '0.6.1', '0.6.2', '0.6.3', '0.6.4', '0.6.5', '0.6.6', '0.6.7',
                '0.6.8', '0.6.9', '0.6.10', '0.6.11', '0.6.12']

    version4 = ['0.7.0', '0.7.1', '0.7.2', '0.7.3', '0.7.4', '0.7.5', '0.7.6']

    version5 = ['0.8.0', '0.8.1', '0.8.2', '0.8.3', '0.8.4', '0.8.5', '0.8.6', '0.8.7', '0.8.8',
                '0.8.9', '0.8.10', '0.8.11', '0.8.12', '0.8.13', '0.8.14', '0.8.15', '0.8.16', '0.8.17']

    if args.language.endswith('0.5'):
        test_versions = version2
        train_versions = version1
    elif args.language.endswith('0.6'):
        test_versions = version3
        train_versions = version1 + version2
    elif args.language.endswith('0.7'):
        test_versions = version4
        train_versions = version1 + version2 + version3
    elif args.language.endswith('0.8'):
        test_versions = version5
        train_versions = version1 + version2 + version3 + version4

    with open('./datasets/{}/compilers.pkl'.format(args.language), 'rb') as fr:
        addr2compiler = pickle.load(fr)

    # addr2compiler = {}
    # with open('./datasets/compiler/addr2compiler.txt', 'r') as fr:
    #     for line in fr.readlines():
    #         addr, version = line.strip().split('\t')
    #         addr2compiler[addr] = version

    addrs = addr2compiler.keys()
    for item in dataset:
        addr = item[5]
        if addr in addrs:
            ver = addr2compiler[addr]
            if ver in test_versions:
                tests.append(item)
            elif ver in train_versions:
                trains.append(item)

    random.shuffle(trains)

    return trains, tests


class TypeDataset(Dataset):
    """
    convert to model input
    """

    def __init__(self, root, data, type2idx, idx2type, args):
        super(TypeDataset, self).__init__()
        self.args = args
        self.original_data = data
        self.type2idx = type2idx
        self.idx2type = idx2type
        self.base_dataset_path = os.path.join(args.base_dataset_path, args.language, args.module)
        self.pos = self.args.pos

        self.vectorize_data = self.convert_features()

    def len(self):
        return len(self.vectorize_data)

    def __getitem__(self, idx):

        item = self.vectorize_data[idx]
        func_id = item['func_id']
        x = torch.from_numpy(np.array(item['cfg_nodes'])).to(torch.float32)
        edge_index = [item['cfg_relations'].row, item['cfg_relations'].col]
        edge_index = torch.from_numpy(np.array(edge_index)).to(torch.long)
        y = torch.tensor(item['label'], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index)
        return data, y, func_id

    def _func2idx(self):

        with open(os.path.join(self.base_dataset_path, 'func_ids.pkl'), 'rb') as fr:
            func_ids_pkl = pickle.load(fr)

        types = func_ids_pkl['func_id_list']

        func2idx = {}
        idx2func = {}

        for idx, ty in enumerate(types):
            func2idx[ty] = idx
            idx2func[idx] = ty

        return func2idx, idx2func

    def convert_features(self):

        vectorize_data = []
        # func2idx, idx2func = self._func2idx()

        for idx in tqdm(range(len(self.original_data))):

            cfg_nodes = self.original_data[idx][0]

            cfg_relations = self.original_data[idx][1]

            f_id = self.original_data[idx][2]
            # if f_id not in func2idx.keys():
            #     continue
            #
            # func_id = func2idx[f_id]

            labels = self.original_data[idx][3]

            param_cnt = self.original_data[idx][4]

            pos = self.pos

            if pos == 'num':
                if param_cnt < 10:
                    vectorize_data.append({
                        'cfg_nodes': cfg_nodes,
                        'cfg_relations': cfg_relations,
                        'label': param_cnt,
                        'func_id': f_id,  # func_id
                    })
            else:
                if param_cnt >= int(pos):
                    label = labels[int(pos) - 1]
                    vectorize_data.append({
                        'cfg_nodes': cfg_nodes,
                        'cfg_relations': cfg_relations,
                        'label': self.type2idx[label],
                        'func_id': f_id,   # func_id
                    })

        logger.info('**** convert features finished!! data size for predicting %s = %s *****' %
                    (self.pos, len(vectorize_data)))

        return vectorize_data

