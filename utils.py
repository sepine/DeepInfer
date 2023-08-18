import os
import time
import torch
from torch.autograd import Variable
from abc import ABC
from torch import nn
import pickle
import numpy as np


def gene_bit2dec():
    bit2dec = {}

    pre = '0x'
    for i in range(32):
        v = (i + 1) * 8
        n = v // 4
        k = pre + ('f' * n)
        bit2dec[k] = str(v)

    return bit2dec


def is_number(s):
    try:
        int(s, 16)
        return True
    except ValueError:
        pass

    return False


def get_classify_types(args):
    types = []
    base_path = os.path.join(args.base_dataset_path, args.language, args.module)

    if args.module == 'return':
        path = os.path.join(base_path, 'vocabs.pkl')
        with open(path, 'rb') as fr:
            types = pickle.load(fr)

    elif args.module == 'param':
        if args.model == 'classify':
            path = os.path.join(base_path, 'vocabs.pkl')
            with open(path, 'rb') as fr:
                vocabs = pickle.load(fr)

            tls = vocabs['type_list']
            for tl in tls:
                if '[' not in tl:
                    types.append(tl)
            types.append('array')

    type2idx = {}
    idx2type = {}

    for idx, ty in enumerate(types):
        type2idx[ty] = idx
        idx2type[idx] = ty

    return type2idx, idx2type


def get_func_id_list(args):
    path = os.path.join(args.base_dataset_path, args.language, args.module)

    with open(os.path.join(path, 'func_ids.pkl'), 'rb') as fr:
        func_ids_pkl = pickle.load(fr)

    func_id_list = func_ids_pkl['func_id_list']

    idx2fid = {}
    fid2idx = {}
    for idx, f_id in enumerate(func_id_list):
        idx2fid[idx] = f_id
        fid2idx[f_id] = idx

    func2addr = func_ids_pkl['func2addr']
    idx2addr = {}
    for f_id in func2addr.keys():
        idx2addr[fid2idx[f_id]] = func2addr[f_id]

    return idx2addr, idx2fid


def save_or_append_pkl(save_path, save_data):
    try:
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
    except (OSError, IOError) as e:
        data = []

    data.append(save_data)

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)


def do_all(args):
    base_path = os.path.join(args.result_path, args.language, args.module)
    save_path = os.path.join(base_path, 'results.pkl')
    with open(save_path, 'rb') as fr:
        data = pickle.load(fr)

    # print(data)

    pred_number = None
    top1 = []
    top3 = []
    top5 = []
    times = []
    numbers = []

    all_t1 = []
    all_t3 = []
    all_t5 = []
    all_times = []
    all_numbers = []

    for idx, item in enumerate(data):
        if idx == 0:
            pred_number = item

        else:
            top1.append(float(item[0][0]['top1_acc']))
            top3.append(float(item[0][0]['top3_acc']))
            top5.append(float(item[0][0]['top5_acc']))
            times.append(float(item[1]))
            numbers.append(int(item[2]))

        all_t1.append(float(item[0][0]['top1_acc']))
        all_t3.append(float(item[0][0]['top3_acc']))
        all_t5.append(float(item[0][0]['top5_acc']))
        all_times.append(float(item[1]))
        all_numbers.append(int(item[2]))

    print("****************** Number Prediction ********************")
    print("Top1 Accuracy: {}".format(pred_number[0][0]['top1_acc']))
    print("Top3 Accuracy: {}".format(pred_number[0][0]['top3_acc']))
    print("Top5 Accuracy: {}".format(pred_number[0][0]['top5_acc']))
    print("Average Prediction Time Cost: {} s".format(pred_number[1]/pred_number[2]))
    print("==============================================")
    print("******************* Type Prediction ********************")
    print("Top1 Accuracy: {}".format(np.average(top1, weights=numbers)))
    print("Top3 Accuracy: {}".format(np.average(top3, weights=numbers)))
    print("Top5 Accuracy: {}".format(np.average(top5, weights=numbers)))
    print("Average Prediction Time Cost: {} s".format(sum(times)/len(times)))
    print("==============================================")

    print("******************* Overall ********************")
    print("Top1 Accuracy: {}".format(np.average(all_t1, weights=all_numbers)))
    print("Top3 Accuracy: {}".format(np.average(all_t3, weights=all_numbers)))
    print("Top5 Accuracy: {}".format(np.average(all_t5, weights=all_numbers)))
    print("Average Prediction Time Cost: {} s".format(sum(all_times) / len(all_times)))
    print("==============================================")


# #################################################################################################
# ################################################################################################

class DecoderBase(ABC, nn.Module):
    def forward(self, encoder_outputs, inputs,
                final_encoder_hidden, targets=None, teacher_forcing=1.0):
        raise NotImplementedError


def to_np(x):
    return x.data.cpu().numpy()


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).contiguous().view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)

    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def trim_seqs(seqs):
    trimmed_seqs = []
    for output_seq in seqs:
        trimmed_seq = []
        for idx in to_np(output_seq):
            trimmed_seq.append(idx[0])
            if idx == 2:
                break

        trimmed_seqs.append(trimmed_seq)

    return trimmed_seqs


def trim_seqs_beam(seqs):
    trimmed_seqs = []
    for output_seq in seqs:
        trimmed_seq = []
        for per_seq in output_seq:
            per_trimmed_seq = []
            for idx in per_seq:
                per_trimmed_seq.append(idx)
                if idx == 2:
                    break
            trimmed_seq.append(per_seq)

        trimmed_seqs.append(trimmed_seq)

    return trimmed_seqs


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed

