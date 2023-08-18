# !/usr/bin/env python
# encoding: utf-8


import os
from tqdm import tqdm
import logging
import random
import numpy as np
import torch
from tqdm import trange
import torch.nn as nn
from datetime import datetime
from utils import to_np, trim_seqs, trim_seqs_beam, get_classify_types, get_func_id_list
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report, top_k_accuracy_score

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def get_input_from_batch(batch, args):
    dfs = batch[0].to(args.device)
    values = batch[1].to(args.device)
    labels = batch[2].to(args.device)
    func_ids = batch[3].to(args.device)

    return dfs, values, labels, func_ids


def trainer(args, model, train_dataset, test_dataset, idx2token, token2idx, schedule, writer):
    # Training model
    start_time = datetime.now()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # logger.info('Num examples = %d', len(train_dataset))

    if args.model == 'gene':

        if not args.test_only:
            train_gene(args, model, train_dataloader, eval_dataloader, token2idx, idx2token, schedule, writer)
            end_time = datetime.now()
            logger.info('Model training spends %s' % str((end_time - start_time).seconds))

        # Test model
        base_path = os.path.join(args.model_save_path, args.language, args.module)
        path = base_path + '/param_pred_data_' + args.model + '_' + args.pos + '_' + \
               str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(args.epochs)
        model.load_state_dict(torch.load(path + '.pt'))

        start_time = datetime.now()
        indicators = decode_evaluate(args, model, eval_dataloader, token2idx, idx2token)
        end_time = datetime.now()

        internal_time = int(str((end_time - start_time).seconds))

        return indicators, internal_time

    elif args.model == 'classify':

        type2idx, idx2type = get_classify_types(args)

        if not args.test_only:

            train_classify(args, model, train_dataloader, eval_dataloader, idx2type)
            end_time = datetime.now()
            logger.info('Model training spends %s' % str((end_time - start_time).seconds))

        # Test model
        base_path = os.path.join(args.model_save_path, args.language, args.module)
        path = base_path + '/param_pred_data_' + args.model + '_' + args.pos + '_' + \
               str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(args.epochs)
        model.load_state_dict(torch.load(path + '.pt'))

        start_time = datetime.now()

        indicators = classify_eval(args, model, eval_dataloader, idx2type)

        end_time = datetime.now()

        internal_time = int(str((end_time - start_time).seconds))

        return indicators, internal_time


def train_classify(args, model, train_dataset, test_dataset, idx2type):

    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    logger.info('****** Running training ******')

    global_step = 0

    model.zero_grad()
    train_iterator = trange(int(args.epochs), desc='Epoch')
    set_seed(args)

    total_loss = 0.
    last_loss = 10000000

    for cur_epoch in train_iterator:

        logger.info('============================= %s =======================' % str(cur_epoch))

        # step = 0
        for batch in tqdm(train_dataset):

            model.train()

            dfs, values, labels, func_ids = get_input_from_batch(batch, args)

            logit = model(dfs, values)

            # Calculate loss for multi-tags
            loss = get_loss_classify(logit, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            total_loss += loss.item()

            optimizer.step()
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step != 1:

                avg_loss = total_loss / global_step

                logger.info('********** Training loss %s ***************', str(avg_loss))

                if avg_loss < last_loss:
                    base_path = os.path.join(args.model_save_path, args.language, args.module)
                    if not os.path.exists(base_path):
                        os.makedirs(base_path)
                    path = base_path + '/param_pred_data_' + args.model + '_' +args.pos + '_' + \
                           str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(args.epochs)
                    torch.save(model.state_dict(), path + '.pt')


def train_gene(args, model, train_dataset, test_dataset, token2idx, idx2token, schedule, writer):

    global_step = 0
    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)

    logger.info('****** Running training ******')

    model.zero_grad()
    set_seed(args)

    total_loss = 0.
    last_loss = 10000000
    best_top1 = 0.

    # for cur_epoch in train_iterator:
    for cur_epoch, teacher_forcing in enumerate(schedule):
        logger.info('============================= %s =======================' % str(cur_epoch))

        for batch in tqdm(train_dataset):

            model.train()

            optimizer.zero_grad()

            dfs, values, labels, func_ids = get_input_from_batch(batch, args)

            output_log_probs, output_seqs = model(dfs, values, targets=labels, teacher_forcing=teacher_forcing)

            batch_size = values.shape[0]
            flattened_outputs = output_log_probs.view(batch_size * args.max_length_output, -1)

            loss = get_loss_gene(flattened_outputs, labels.contiguous().view(-1), token2idx)
            loss.backward()

            total_loss += loss.item()

            optimizer.step()

            if global_step > 0:

                base_path = os.path.join(args.model_save_path, args.language, args.module)

                if not os.path.exists(base_path):
                    os.makedirs(base_path)

                path = base_path + '/param_pred_data_' + args.model + '_' + args.pos + '_' + \
                       str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(args.epochs)

                torch.save(model.state_dict(), path + '.pt')

            global_step += 1

        print('-' * 100, flush=True)


def get_loss_gene(logits, y_true, token2idx):
    loss = nn.NLLLoss(ignore_index=token2idx['<PAD>'])
    output = loss(logits, y_true)
    return output


def get_loss_classify(logits, y_true):
    loss = nn.CrossEntropyLoss()   # nn.NLLLoss()
    output = loss(logits, y_true)
    return output


def decode_evaluate(args, model, eval_dataset, token2idx, idx2token):
    print('evaluate model by beam search')

    # Eval
    logger.info('****** Running evaluation *******')
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    trues = None
    func_id_lists = None
    for batch in tqdm(eval_dataset):

        model.eval()

        with torch.no_grad():

            dfs, values, labels, func_ids = get_input_from_batch(batch, args)

            decoded_batch = model.decode(dfs, values, labels)

            batch_outputs_ids = trim_seqs_beam(decoded_batch)

            batch_targets_ids = [list(seq[seq > 0]) for seq in list(to_np(labels))]

            batch_outputs = convert2type_topk(args, batch_outputs_ids, idx2token)
            batch_targets = convert2type(batch_targets_ids, idx2token, args.default_vocab_len)

        nb_eval_steps += 1

        if preds is None:
            preds = batch_outputs
            trues = batch_targets
            func_id_lists = func_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, batch_outputs, axis=0)
            trues = np.append(trues, batch_targets, axis=0)
            func_id_lists = np.append(func_id_lists, func_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    indicators = gene_indicator(preds, trues, func_id_lists, args)

    return indicators


def convert2type_topk(args, output_ids, idx2token):
    out_lists = []
    for per_output_ids in output_ids:
        per_out_list = []
        for ids in per_output_ids:
            tmp_out = ''
            for i in ids:
                token = idx2token[i]
                if token == '<SOS>':
                    continue
                elif token == '<EOS>':
                    break
                elif token == '<BLK>':
                    tmp_out += '[]'
                else:
                    if int(i) >= args.default_vocab_len:
                        tmp_out += '[' + token + ']'
                    else:
                        tmp_out += token
            per_out_list.append(tmp_out)
        out_lists.append(per_out_list)
    return out_lists


def convert2type(output_ids, idx2token, default_vocab_len):
    out_list = []
    for ids in output_ids:
        tmp_out = ''
        for i in ids:
            token = idx2token[i]
            if token == '<SOS>':
                continue
            elif token == '<EOS>':
                break
            elif token == '<BLK>':
                tmp_out += '[]'
            else:
                if int(i) >= default_vocab_len:
                    tmp_out += '[' + token + ']'
                else:
                    tmp_out += token
        out_list.append(tmp_out)
    return out_list


def gene_indicator(preds, trues, func_id_lists, args):

    func_id_pred_trues = []

    assert len(preds) == len(trues)

    top1_acc = cal_top_k_acc(preds, trues, topk=1)
    top3_acc = cal_top_k_acc(preds, trues, topk=3)
    top5_acc = cal_top_k_acc(preds, trues, topk=5)

    idx2addr, idx2fid = get_func_id_list(args)

    for i in range(len(preds)):
        func_id_pred_trues.append((list(preds[i]), trues[i],
                                   idx2fid[int(func_id_lists[i])], idx2addr[int(func_id_lists[i])]))

    ret = [{
        'top1_acc': str(top1_acc),
        'top3_acc': str(top3_acc),
        'top5_acc': str(top5_acc),
    }]

    return ret


def cal_top_k_acc(preds, trues, topk):
    cnt = 0
    for i in range(len(preds)):
        topk_preds = preds[i][:topk]
        if trues[i] in topk_preds:
            cnt += 1

    acc = cnt / len(preds)
    return acc


def classify_eval(args, model, eval_dataset, idx2type):
    print('evaluate model')

    # Eval
    logger.info('****** Running evaluation *******')

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    func_id_lists = None
    for batch in tqdm(eval_dataset):
        model.eval()

        with torch.no_grad():

            dfs, values, labels, func_ids = get_input_from_batch(batch, args)

            logits = model(dfs, values)

            tmp_eval_loss = get_loss_classify(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
            func_id_lists = func_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            func_id_lists = np.append(func_id_lists, func_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    indicators = classify_indicator(preds, out_label_ids, func_id_lists, idx2type, args)

    return indicators


def classify_indicator(logits, y_true, func_id_lists, idx2type, args):
    ret = []
    func_id_pred_trues = []

    y_pred = np.argmax(logits, axis=1)

    top1_acc = top_k_accuracy_score(y_true, logits, k=1, labels=range(0, args.num_classes))
    top3_acc = top_k_accuracy_score(y_true, logits, k=3, labels=range(0, args.num_classes))
    top5_acc = top_k_accuracy_score(y_true, logits, k=5, labels=range(0, args.num_classes))

    idx2addr, idx2fid = get_func_id_list(args)

    for i in range(len(y_pred)):
        func_id_pred_trues.append((idx2type[y_pred[i]], idx2type[y_true[i]],
                                   idx2fid[int(func_id_lists[i])], idx2addr[int(func_id_lists[i])]))

    ret.append({
        'top1_acc': str(top1_acc),
        'top3_acc': str(top3_acc),
        'top5_acc': str(top5_acc),
    })

    return ret
