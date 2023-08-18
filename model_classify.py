# !/usr/bin/env python
# encoding: utf-8


import logging
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

logger = logging.getLogger(__name__)


class ReturnTypeInference(nn.Module):
    def __init__(self, args):
        super(ReturnTypeInference, self).__init__()

        self.args = args
        self.dropout_rate = self.args.dropout_rate
        self.embedding_dim = self.args.embedding_dim
        self.hidden_size = self.args.hidden_size
        self.num_gnn_layers = self.args.num_gnn_layers
        self.num_classes = self.args.num_classes

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.gat = GAT(self.embedding_dim, self.hidden_size, self.hidden_size, heads=1)
        self.attn_cfg = GraphLocalAttention(args.hidden_size)

        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, cfg_nodes, cfg_relations, node_pos):
        cfg_nodes = self.gat(cfg_nodes, cfg_relations)
        x = self.attn_by_batch(cfg_nodes, node_pos, self.args)
        x = self.linear(x)

        return x

    def mean_by_batch(self, cfg_nodes, node_pos, args):
        cfg = torch.zeros((len(node_pos) - 1, self.hidden_size)).to(args.device)
        for idx in range(len(node_pos) - 1):
            cur = cfg_nodes[node_pos[idx]: node_pos[idx + 1], :]
            cfg[idx, :] = torch.mean(cur, dim=0)

        return cfg

    def attn_by_batch(self, cfg_nodes, node_pos, args):
        cfg = torch.zeros((len(node_pos) - 1, self.hidden_size)).to(args.device)
        for idx in range(len(node_pos) - 1):
            cur = cfg_nodes[node_pos[idx]: node_pos[idx + 1], :]
            attn_cur, _ = self.attn_cfg(cur)
            cfg[idx, :] = attn_cur
        return cfg


class BasicTypeClassifier(nn.Module):
    def __init__(self, vocab2vec, word2idx, args):
        super(BasicTypeClassifier, self).__init__()

        self.vocab2vec = vocab2vec
        self.word2idx = word2idx

        self.args = args
        self.dropout_rate = self.args.dropout_rate
        self.embedding_dim = self.args.embedding_dim
        self.hidden_size = self.args.hidden_size
        self.num_gnn_layers = self.args.num_gnn_layers
        self.num_classes = self.args.num_classes

        self.const_emb = nn.Embedding(len(word2idx.keys()), self.embedding_dim)
        self.const_emb.weight.data.normal_(0, 1 / self.embedding_dim ** 0.5)

        self.hierarchical_attn = HierarchicalAttention(self.vocab2vec, self.args)

        self.const_map = nn.Linear(self.embedding_dim, self.hidden_size)

        self.attn = GlobalAttention(args.hidden_size)

        self.attn_const = GlobalAttention(args.hidden_size)

        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, dfs_seq, const_values):
        dfs = self.hierarchical_attn(dfs_seq)

        const = self.proc_const(const_values)

        x, _ = self.attn(torch.stack([dfs, const], dim=1))

        x = self.linear(x)

        return x

    def proc_const(self, const_values):
        c_emb = self.const_emb(const_values)
        c_emb = self.const_map(c_emb)
        c_emb, _ = self.attn_const(c_emb)
        return c_emb


class NumInference(nn.Module):
    def __init__(self, vocab2vec, word2idx, args):
        super(NumInference, self).__init__()

        self.vocab2vec = vocab2vec
        self.word2idx = word2idx

        self.args = args

        self.dropout_rate = self.args.dropout_rate
        self.embedding_dim = self.args.embedding_dim
        self.hidden_size = self.args.hidden_size
        self.num_gnn_layers = self.args.num_gnn_layers
        self.num_classes = self.args.num_classes

        self.const_emb = nn.Embedding(len(word2idx.keys()), self.embedding_dim)
        self.const_emb.weight.data.normal_(0, 1 / self.embedding_dim ** 0.5)

        self.hierarchical_attn = HierarchicalAttention(self.vocab2vec, self.args)

        self.const_map = nn.Linear(self.embedding_dim, self.hidden_size)

        self.attn = GlobalAttention(args.hidden_size)

        self.attn_const = GlobalAttention(args.hidden_size)

        self.linear = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, dfs_seq, const_values):

        dfs = self.hierarchical_attn(dfs_seq)

        const = self.proc_const(const_values)

        x, _ = self.attn(torch.stack([dfs, const], dim=1))

        x = self.linear(x)

        return x

    def proc_const(self, const_values):
        c_emb = self.const_emb(const_values)
        c_emb = self.const_map(c_emb)
        c_emb, _ = self.attn_const(c_emb)
        return c_emb


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)

        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphLocalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 5),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 5, 1))

    def forward(self, encoder_outputs):

        energy = self.projection(encoder_outputs)
        energy = energy.squeeze(-1)

        weights = F.softmax(energy, dim=0)

        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=0)
        return outputs, weights


class GlobalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 5),
            nn.ReLU(True),
            nn.Linear(hidden_dim // 5, 1))

    def forward(self, encoder_outputs):

        energy = self.projection(encoder_outputs)
        energy = energy.squeeze(-1)

        weights = F.softmax(energy, dim=1)

        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class HierarchicalAttention(nn.Module):
    def __init__(self, vocab2vec, args):
        super(HierarchicalAttention, self).__init__()
        self.vocab2vec = vocab2vec
        self.args = args
        self.embedding_dim = self.args.embedding_dim
        self.hidden_size = self.args.hidden_size
        self.rnn_layers = self.args.rnn_layers
        self.dropout_rate = self.args.dropout_rate

        self.dfs_attn = DFSAttention(self.vocab2vec, self.embedding_dim,
                                     self.hidden_size, self.rnn_layers,
                                     self.dropout_rate, self.args)

        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=self.rnn_layers,
                           bidirectional=True, batch_first=True, dropout=self.dropout_rate)
        self.global_attn = GlobalAttention(args.hidden_size)

        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, dfs_seq_list):

        output_list = []
        dfs_seq_list = dfs_seq_list.permute(1, 0, 2)
        for dfs in dfs_seq_list:
            output = self.dfs_attn(dfs)
            output_list.append(output)

        output = torch.stack(output_list, dim=1)

        output, _ = self.rnn(output)

        query = self.dropout(output)
        attn_output, attn = self.attn_net(output, query)

        logit = self.fc(attn_output)
        logit = self.dropout(logit)

        return logit

    def attn_net(self, x, query, mask=None):

        d_k = query.size(-1)

        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)

        p_attn = F.softmax(scores, dim=-1)

        context = torch.matmul(p_attn, x).sum(1)

        return context, p_attn


class DFSAttention(nn.Module):
    def __init__(self, vocab2vec, embedding_dim, hidden_size, rnn_layers, dropout_rate, args):
        super(DFSAttention, self).__init__()

        self.vocab2vec = vocab2vec
        self.args = args
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout_rate = dropout_rate

        num_embed = len(self.vocab2vec)
        pretrained_dict = self.vocab2vec
        pretrained_dict[1, :] = np.zeros(self.embedding_dim)
        pretrained_dict = torch.from_numpy(pretrained_dict)

        self.emb = nn.Embedding(num_embeddings=num_embed,
                                embedding_dim=self.embedding_dim,
                                dtype=torch.float32).from_pretrained(pretrained_dict).float()

        self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.rnn_layers,
                           bidirectional=True, batch_first=True, dropout=self.dropout_rate)

        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs):

        inputs = self.emb(inputs)

        output, _ = self.rnn(inputs)

        query = self.dropout(output)
        attn_output, attn = self.attn_net(output, query)  #

        logit = self.fc(attn_output)

        return logit

    def attn_net(self, x, query, mask=None):

        d_k = query.size(-1)

        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)

        p_attn = F.softmax(scores, dim=-1)

        context = torch.matmul(p_attn, x).sum(1)

        return context, p_attn

