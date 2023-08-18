# !/usr/bin/env python
# encoding: utf-8


import os
from tqdm import tqdm
import numpy as np
import pickle
import logging
import torch
import random
from utils import gene_bit2dec, is_number, get_classify_types
from torch_geometric.data import Dataset

logger = logging.getLogger(__name__)


def load_datasets_and_vocabs(args):

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
            train_dataset, test_dataset, opid2vec, opcode2idx, idx2opcode, token2idx, idx2token, default_vocab_len = pickle.load(fr)
        # logger.info('Load data from caches. The total number of train dataset is %d' % len(train_dataset))
        # logger.info('Load data from caches. The total number of test dataset is %d' % len(test_dataset))
        return train_dataset, test_dataset, opid2vec, opcode2idx, idx2opcode, token2idx, idx2token, default_vocab_len
    # ##############################  Cache Data ########################################

    with open(os.path.join(dataset_path, 'param_data.pkl'), 'rb') as fr:
        param_dataset = pickle.load(fr)

    opid2vec, opcode2idx, idx2opcode, token2idx, idx2token, default_vocab_len = get_vocab_vec(args)

    # # split train and test dataset
    if args.language.startswith('compiler'):
        train, test = split_train_test_compiler(param_dataset, args)
    else:
        total_cnt = len(param_dataset)
        split_cnt = (total_cnt // 5) * 4
        train = param_dataset[:split_cnt]
        random.shuffle(train)

        test = param_dataset[split_cnt:]

    train_dataset = TypeDataset(args.base_dataset_path, train, opcode2idx, token2idx, args)
    test_dataset = TypeDataset(args.base_dataset_path, test, opcode2idx, token2idx, args)

    # logger.info('Build Train finished! The total number is %d' % len(train_dataset))
    # logger.info('Build Test finished! The total number is %d' % len(test_dataset))

    # ##############################  Cache Data ########################################

    with open(cached_data_path, 'wb') as fw:
        pickle.dump((train_dataset, test_dataset, opid2vec,
                     opcode2idx, idx2opcode, token2idx, idx2token, default_vocab_len), fw)

    # logger.info('Save data to caches successfully. The total number of train dataset is %d' % len(train_dataset))
    # logger.info('Save data to caches successfully. The total number of test dataset is %d' % len(test_dataset))

    # ##############################  Cache Data ########################################

    return train_dataset, test_dataset, opid2vec, opcode2idx, idx2opcode, token2idx, idx2token, default_vocab_len


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

    # versions = ['0.1.1', '0.1.2', '0.1.3', '0.1.4', '0.1.5', '0.1.6', '0.1.7',
    #             '0.2.0', '0.2.1', '0.2.2',
    #             '0.3.0', '0.3.1', '0.3.2', '0.3.3', '0.3.4', '0.3.5', '0.3.6',
    #             '0.4.0', '0.4.1', '0.4.2', '0.4.3', '0.4.4', '0.4.5', '0.4.6', '0.4.7', '0.4.8', '0.4.9', '0.4.10',
    #             '0.4.11', '0.4.12', '0.4.13', '0.4.14', '0.4.15', '0.4.16', '0.4.17', '0.4.18', '0.4.19', '0.4.20',
    #             '0.4.21', '0.4.22', '0.4.23', '0.4.24', '0.4.25', '0.4.26',
    #             '0.5.0', '0.5.1', '0.5.2', '0.5.3', '0.5.4', '0.5.5', '0.5.6', '0.5.7', '0.5.8',
    #             '0.5.9', '0.5.10', '0.5.11', '0.5.12', '0.5.13', '0.5.14', '0.5.15', '0.5.16', '0.5.17',
    #             '0.6.0', '0.6.1', '0.6.2', '0.6.3', '0.6.4', '0.6.5', '0.6.6', '0.6.7',
    #             '0.6.8', '0.6.9', '0.6.10', '0.6.11', '0.6.12',
    #             '0.7.0', '0.7.1', '0.7.2', '0.7.3', '0.7.4', '0.7.5', '0.7.6',
    #             '0.8.0', '0.8.1', '0.8.2', '0.8.3', '0.8.4', '0.8.5', '0.8.6', '0.8.7', '0.8.8',
    #             '0.8.9', '0.8.10', '0.8.11', '0.8.12', '0.8.13', '0.8.14', '0.8.15', '0.8.16', '0.8.17']

    # test_versions = ['0.8.0', '0.8.1', '0.8.2', '0.8.3', '0.8.4', '0.8.5', '0.8.6', '0.8.7', '0.8.8',
    #                  '0.8.9', '0.8.10', '0.8.11', '0.8.12', '0.8.13', '0.8.14', '0.8.15', '0.8.16', '0.8.17']

    # test_versions = ['0.7.0', '0.7.1', '0.7.2', '0.7.3', '0.7.4', '0.7.5', '0.7.6']

    # test_versions = ['0.6.0', '0.6.1', '0.6.2', '0.6.3', '0.6.4', '0.6.5', '0.6.6', '0.6.7',
    #                  '0.6.8', '0.6.9', '0.6.10', '0.6.11', '0.6.12']

    # test_versions = ['0.5.0', '0.5.1', '0.5.2', '0.5.3', '0.5.4', '0.5.5', '0.5.6', '0.5.7', '0.5.8']
    #
    # train_versions = ['0.1.1', '0.1.2', '0.1.3', '0.1.4', '0.1.5', '0.1.6', '0.1.7',
    #                   '0.2.0', '0.2.1', '0.2.2',
    #                   '0.3.0', '0.3.1', '0.3.2', '0.3.3', '0.3.4', '0.3.5', '0.3.6',
    #                   '0.4.0', '0.4.1', '0.4.2', '0.4.3', '0.4.4', '0.4.5', '0.4.6', '0.4.7', '0.4.8', '0.4.9', '0.4.10',
    #                   '0.4.11', '0.4.12', '0.4.13', '0.4.14', '0.4.15', '0.4.16', '0.4.17', '0.4.18', '0.4.19', '0.4.20',
    #                   '0.4.21', '0.4.22', '0.4.23', '0.4.24', '0.4.25', '0.4.26']

    # with open(save_path, 'wb') as fw:
    #     pickle.dump(addr2compiler, fw)
    # addr2compiler = {}
    with open('./datasets/{}/compilers.pkl'.format(args.language), 'rb') as fr:
        # for line in fr.readlines():
        #     addr, version = line.strip().split('\t')
        #     addr2compiler[addr] = version
        addr2compiler = pickle.load(fr)

    addrs = addr2compiler.keys()
    for item in dataset:
        addr = item[3]
        if addr in addrs:
            ver = addr2compiler[addr]
            if ver in test_versions:
                tests.append(item)
            elif ver in train_versions:
                trains.append(item)

    random.shuffle(trains)

    return trains, tests


def get_vocab_vec(args):

    opcode2vec = {}
    with open(os.path.join(
            args.base_dataset_path, 'word2vec/word2vec_' + str(args.language) + '_' + str(args.module) + '.vec'), 'r') as fr:

        for line in fr.readlines():
            line = line.strip().split(' ')
            if len(line) > 1:
                opcode2vec[line[0]] = np.array(line[1:], dtype=np.float64)

    opcodes = [op for op in opcode2vec.keys()]

    opcode2idx, idx2opcode = {}, {}
    for idx, op in enumerate(opcodes):
        opcode2idx[op] = idx
        idx2opcode[idx] = op

    opid2vec = []
    for idx in idx2opcode.keys():
        opid2vec.append(opcode2vec[idx2opcode[idx]])

    vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', '<BLK>', '<SEM>']

    default_vocab = []

    path = os.path.join(args.base_dataset_path, args.language, args.module)
    with open(os.path.join(path, 'vocabs.pkl'), 'rb') as fr:
        vocabs = pickle.load(fr)

    if args.model == 'gene':
        default_vocab = vocabs['default']

    vocab = vocab + default_vocab

    default_vocab_len = len(vocab)

    const_vocab = []

    if args.module == 'param':
        const_vocab = vocabs['const']

    vocab = vocab + const_vocab

    token2idx, idx2token = {}, {}
    for idx, word in enumerate(vocab):
        token2idx[word] = idx
        idx2token[idx] = word

    return np.array(opid2vec).astype(np.float), opcode2idx, idx2opcode, token2idx, idx2token, default_vocab_len


class TypeDataset(Dataset):
    """
    convert to model input
    """

    def __init__(self, root, data, opcode2idx, token2idx, args):
        super(TypeDataset, self).__init__(root, data, token2idx, args)
        self.args = args
        self.original_data = data
        self.opcode2idx = opcode2idx
        self.word2idx = token2idx
        self.base_dataset_path = os.path.join(args.base_dataset_path, args.language, args.module)
        self.pos = self.args.pos

        self.bit2vec = gene_bit2dec()
        self.vectorize_data = self.convert_features()

    def len(self):
        return len(self.vectorize_data)

    def __getitem__(self, idx):

        item = self.vectorize_data[idx]

        func_id = item['func_id']

        y = torch.tensor(item['label'], dtype=torch.long)
        dfs_features = torch.from_numpy(item['dfs_seq']).to(torch.long)
        const_values = torch.from_numpy(item['const_values']).to(torch.long)

        return dfs_features, const_values, y, func_id

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
        func2idx, idx2func = self._func2idx()

        type2idx, idx2type = get_classify_types(args=self.args)

        for idx in tqdm(range(len(self.original_data))):

            func_id = func2idx[self.original_data[idx][0]]

            param_cnt = self.original_data[idx][2]

            vectorized_dfs = self.original_data[idx][4]
            vectorized_dfs = self._proc_dfs(vectorized_dfs)

            const_values = self.original_data[idx][5]
            const_values = self._proc_const_values(const_values)

            labels = self.original_data[idx][1]

            pos = self.pos
            if pos == 'num':
                if param_cnt < 10:
                    vectorize_data.append({
                        'func_id': func_id,
                        'dfs_seq': vectorized_dfs,
                        'const_values': const_values,
                        'label': param_cnt
                    })
            else:
                if self.args.model == 'classify':
                    if param_cnt >= int(pos):
                        label = labels[int(pos) - 1]
                        if 'storage' in label:
                            label = 'slot'
                        if '[' in label:
                            label = 'array'

                        label = type2idx[label]
                        vectorize_data.append({
                            'func_id': func_id,
                            'dfs_seq': vectorized_dfs,
                            'const_values': const_values,
                            'label': label
                        })
                elif self.args.model == 'gene':
                    if param_cnt >= int(pos):
                        label = labels[int(pos) - 1]
                        if 'storage' in label:
                            continue
                        if '[' in label:
                            label = self._proc_label_gene(label)
                            vectorize_data.append({
                                'func_id': func_id,
                                'dfs_seq': vectorized_dfs,
                                'const_values': const_values,
                                'label': label
                            })

        logger.info('**** convert features finished!! data size for predicting %s = %s *****' %
                    (self.pos, len(vectorize_data)))

        return vectorize_data

    def _proc_dfs(self, dfs):
        seq_dfs_list = []
        for per_d in dfs:
            per_dfs = []
            for op in per_d:
                per_dfs.append(self.opcode2idx[op])
            if len(per_dfs) > 0:
                seq_dfs_list.append(self._pad_dfs(per_dfs))

        padded_dfs_list = self._pad_dfs_list(seq_dfs_list)
        padded_dfs_list = [per_dfs[: self.args.max_length_word] for per_dfs in padded_dfs_list][: self.args.max_length_dfs]

        padded_dfs_list = np.stack(arrays=padded_dfs_list, axis=0)
        padded_dfs_list = padded_dfs_list.astype(np.int64)

        return padded_dfs_list

    def _pad_dfs_list(self, dfs_list):
        padded_dfs_list = []
        for dfs in dfs_list:
            padded_dfs = self._pad_dfs(dfs)
            padded_dfs_list.append(padded_dfs)

        while len(padded_dfs_list) < self.args.max_length_dfs:
            padded_dfs_list.append([int(self.opcode2idx['<PAD>'])] * self.args.max_length_word)

        return padded_dfs_list

    def _pad_dfs(self, dfs):

        if len(dfs) < self.args.max_length_word:
            dfs = dfs + [int(self.opcode2idx['<PAD>'])] * (self.args.max_length_word - len(dfs))

        return dfs

    def _proc_const_values(self, const_values):
        const_emb = []

        for per_cv in const_values:
            for val in per_cv:
                if val in self.word2idx.keys():
                    conv_val = self._convert_to_num(val)
                    if conv_val is not None:
                        const_emb.append(self.word2idx[conv_val])

        const_emb = list(set(const_emb))

        if len(const_emb) < self.args.max_length_const:
            const_emb = const_emb + [self.word2idx['<PAD>']] * (self.args.max_length_const - len(const_emb))

        const_emb = np.array(const_emb, dtype=np.long)

        return const_emb

    def _convert_to_num(self, val):

        if is_number(val) and int(val, 16) <= 255:
            return str(int(val, 16))
        else:
            if val in self.bit2vec.keys():
                return self.bit2vec[val]
            else:
                return None

    def _proc_label_gene(self, label):
        ret = [self.word2idx['<SOS>']]
        split_label = label.split('[')

        prefix = split_label[0]
        ret = ret + [self.word2idx[prefix]]

        if len(split_label) > 1:
            for i in range(1, len(split_label)):
                if len(split_label[i]) == 1:
                    ret.append(self.word2idx['<BLK>'])
                else:
                    ret.append(self.word2idx[split_label[i][:-1]])

        ret.append(self.word2idx['<EOS>'])
        if len(ret) < self.args.max_length_output:
            ret = ret + [int(self.word2idx['<PAD>'])] * (self.args.max_length_output - len(ret))

        return np.array(ret, dtype=np.long)

