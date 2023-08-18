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
    node_pos = batch[0].ptr.to(args.device)
    x = batch[0].x.to(args.device)
    edge_index = batch[0].edge_index.to(args.device)
    labels = batch[1].to(args.device)
    func_ids = batch[2]

    return x, edge_index, node_pos, labels, func_ids


def trainer(args, model, train_dataset, test_dataset, type2idx, idx2type):
    # Training model
    start_time = datetime.now()
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    # logger.info('Num examples = %d', len(train_dataset))

    if args.model == 'classify':

        if not args.test_only:

            train_classify(args, model, train_dataloader, eval_dataloader, idx2type)

            end_time = datetime.now()
            logger.info('Model training spends %s' % str((end_time - start_time).seconds))

        # Test model
        base_path = os.path.join(args.model_save_path, args.language, args.module)
        path = base_path + '/return_pred_data_' + args.model + '_' + args.pos + '_' + \
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

    best_top1 = 0.

    for cur_epoch in train_iterator:

        logger.info('============================= %s =======================' % str(cur_epoch))

        # step = 0
        for batch in tqdm(train_dataset):

            model.train()

            x, edge_index, node_pos, labels, func_ids = get_input_from_batch(batch, args)

            logit = model(x, edge_index, node_pos)

            # Calculate loss for multi-tags
            loss = get_loss_classify(logit, labels)

            # if args.gradient_accumulation_steps > 1:
            #     loss = loss / args.gradient_accumulation_steps

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)

            total_loss += loss.item()

            optimizer.step()
            model.zero_grad()
            global_step += 1

            # if args.logging_steps > 0 and global_step != 1 and global_step % args.logging_steps == 0:
            if args.logging_steps > 0 and global_step != 1:

                avg_loss = total_loss / global_step

                logger.info('********** Training loss %s ***************', str(avg_loss))

                if avg_loss < last_loss:
                    # save model checkpoint according to the loss
                    base_path = os.path.join(args.model_save_path, args.language, args.module)
                    if not os.path.exists(base_path):
                        os.makedirs(base_path)
                    path = base_path + '/return_pred_data_' + args.model + '_' + args.pos + '_' + \
                           str(args.batch_size) + '_' + str(args.learning_rate) + '_' + str(args.epochs)
                    torch.save(model.state_dict(), path + '.pt')


def get_loss_classify(logits, y_true):
    loss = nn.CrossEntropyLoss()
    output = loss(logits, y_true)
    return output


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

            x, edge_index, node_pos, labels, func_ids = get_input_from_batch(batch, args)

            logits = model(x, edge_index, node_pos)

            tmp_eval_loss = get_loss_classify(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1

        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
            if args.language == 'solidity':
                func_id_lists = func_ids.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)
            if args.language == 'solidity':
                func_id_lists = np.append(func_id_lists, func_ids.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps

    logger.info('********** Current loss %s ***************', str(eval_loss))

    indicators = classify_indicator(preds, out_label_ids, func_id_lists, idx2type, args)

    return indicators


def classify_indicator(logits, y_true, func_id_lists, idx2type, args):
    ret = []
    func_id_pred_trues = []

    y_pred = np.argmax(logits, axis=1)

    top1_acc = top_k_accuracy_score(y_true, logits, k=1, labels=range(0, args.num_classes))
    top3_acc = top_k_accuracy_score(y_true, logits, k=3, labels=range(0, args.num_classes))
    top5_acc = top_k_accuracy_score(y_true, logits, k=5, labels=range(0, args.num_classes))

    ret.append({
        'top1_acc': str(top1_acc),
        'top3_acc': str(top3_acc),
        'top5_acc': str(top5_acc),
    })

    return ret