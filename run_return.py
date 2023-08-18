# !/usr/bin/env python
# encoding: utf-8

import os

import torch
import numpy as np
import random
import argparse
import logging
import json
from data_loader_return import load_datasets_and_vocabs
from model_classify import ReturnTypeInference
from trainer_return import trainer
from datetime import datetime
from utils import get_classify_types, save_or_append_pkl, do_all


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for initialization')

    # ################ Fixed Parameters ###########################
    parser.add_argument('--base_dataset_path', type=str, default='datasets',
                        help='base dataset path')

    parser.add_argument('--cached_dataset_path', type=str, default='cached',
                        help='cached dataset path')

    parser.add_argument('--result_path', type=str, default='results',
                        help='cache dataset path')

    parser.add_argument('--model_save_path', type=str, default='models',
                        help='Path to models.')

    parser.add_argument('--max_length_word', type=int, default=128,
                        help='max number of opcodes in each dataflow sequence')

    parser.add_argument('--max_length_dfs', type=int, default=32,
                        help='max number of dataflow sequences')

    parser.add_argument('--max_length_const', type=int, default=64,
                        help='max number of const in each dataflow sequence')

    parser.add_argument('--max_length_output', type=int, default=7,
                        help='max number of output sequences')

    parser.add_argument('--teacher_forcing_fraction', type=float, default=0.5,
                        help='fraction of batches that will use teacher forcing during training')

    parser.add_argument('--scheduled_teacher_forcing', type=bool, default=True,
                        # action='store_true',
                        help='Linearly decrease the teacher forcing fraction '
                             'from 1.0 to 0.0 over the specified number of epochs')

    parser.add_argument('--test_only', type=bool, default=False,
                        help='fraction of batches that will use teacher forcing during training')

    parser.add_argument('--print_all', type=bool, default=False,
                        help='fraction of batches that will use teacher forcing during training')

    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every ... update steps')

    # ################ Fixed Parameters ###########################

    # ################ Hyper-parameters ###########################

    parser.add_argument('--pos', type=str, default='1',
                        choices=['num', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
                        help='choose training mode')

    parser.add_argument('--language', type=str, default='solidity',
                        choices=['solidity', 'vyper', 'compiler0.5', 'compiler0.6', 'compiler0.7', 'compiler0.8'],
                        help='choose the language')

    parser.add_argument('--model', type=str, default='classify',
                        choices=['classify'],
                        help='which type of the model should we select')

    parser.add_argument('--module', type=str, default='return',
                        choices=['return'],
                        help='which module should we select')

    parser.add_argument("--batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--test_batch_size", default=128, type=int,
                        help="Batch size per GPU/CPU for testing.")

    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam. classify = 1e-4; gene = 1e-4")

    parser.add_argument('--epochs', type=int, default=50,
                        help='Total number of training epochs to perform')

    # parser.add_argument('--num_classes', type=int, default=2,
    #                     help='Number of classes')

    parser.add_argument('--embedding_dim', type=int, default=100,
                        help='Dimension of word2vec embeddings')

    parser.add_argument('--hidden_size', type=int, default=200,
                        help='Dimension of hidden layers')

    parser.add_argument('--num_gnn_layers', type=int, default=2,
                        help='Number of GNN layers')

    parser.add_argument('--rnn_layers', type=int, default=2,
                        help='Number of RNN layers')

    parser.add_argument('--decoder_type', type=str, default='copy',
                        help='Type of Decoder layer')

    parser.add_argument('--beam_width', type=int, default=10,
                        help='Max width of each beam')

    parser.add_argument('--topk', type=int, default=5,
                        help='Top k output')

    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help='dropout rate')

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--keep_prob', type=float, default=1.0,
                        help='Probablity of keeping an element in the dropout step.')

    # ################ Hyper-parameters ###########################

    return parser.parse_args()


def check_args(args):
    logger.info(vars(args))


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # Parse args
    args = parse_args()

    if args.print_all:
        do_all(args)
        exit()

    # Process classify number
    if args.pos == 'num':
        type2idx, idx2type = get_classify_types(args)
        args.num_classes = 10
    else:
        if args.model == 'classify':
            type2idx, idx2type = get_classify_types(args)
            args.num_classes = len(type2idx)

    # check_args(args)

    # Setup CUDA, GPU training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    # Load datasets
    start_time = datetime.now()
    train_dataset, test_dataset = load_datasets_and_vocabs(args, type2idx, idx2type)
    end_time = datetime.now()
    logger.info('Load dataset spends %s' % str((end_time - start_time).seconds))
    args.test_dataset_length = len(test_dataset)

    # Build Model
    model = ReturnTypeInference(args)

    model.to(args.device)

    if args.model == 'classify':
        # Train and test
        indicators, internal_time = trainer(args, model, train_dataset, test_dataset, type2idx, idx2type)

        base_path = os.path.join(args.result_path, args.language, args.module)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        save_path = os.path.join(base_path, 'results.pkl')

        save_data = (indicators, internal_time / args.test_dataset_length, args.test_dataset_length)

        save_or_append_pkl(save_path, save_data)

    print('Save results successfully! ')


if __name__ == '__main__':
    main()
