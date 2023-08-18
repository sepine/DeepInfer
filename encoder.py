# !/usr/bin/env python
# encoding: utf-8


import logging
import torch
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TypeEncoder(nn.Module):
    def __init__(self, opid2vec, opcode2idx, word2idx, hidden_size, embedding_dim, args):
        super(TypeEncoder, self).__init__()

        # Hyper Parameters
        self.args = args
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.opid2vec = opid2vec
        self.opcode2idx = opcode2idx

        # Initial Embedding
        self.const_emb = nn.Embedding(len(word2idx.keys()), self.embedding_dim)
        self.const_emb.weight.data.normal_(0, 1 / self.embedding_dim ** 0.5)

        # Hierarchical Attention for data flow infos
        self.dfs_h_attn = HierarchicalAttention(self.opid2vec, self.opcode2idx, self.args)

        # Const process
        self.const_map = nn.Linear(self.embedding_dim, self.hidden_size)

    def forward(self, dfs_seq, values):

        dfs = self.dfs_h_attn(dfs_seq)   # B x D

        sem = dfs

        const = self.const_emb(values)       # B x L x D
        const = self.const_map(const)    # B x L x D

        enc_out = torch.cat([sem.unsqueeze(1), const], dim=1)   # B x L + 1 x D

        return enc_out

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

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(2, batch_size, self.hidden_size))  # bidirectional rnn
        if next(self.parameters()).is_cuda:
            return hidden.cuda()
        else:
            return hidden


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
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
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        energy = energy.squeeze(-1)
        weights = F.softmax(energy, dim=0)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=0)
        return outputs, weights


class HierarchicalAttention(nn.Module):
    def __init__(self, opid2vec, opcode2idx, args):
        super(HierarchicalAttention, self).__init__()
        self.opid2vec = opid2vec
        self.opcode2idx = opcode2idx

        self.args = args

        self.embedding_dim = self.args.embedding_dim
        self.hidden_size = self.args.hidden_size
        self.rnn_layers = self.args.rnn_layers
        self.dropout_rate = self.args.dropout_rate

        self.dfs_attn = DFSAttention(self.opid2vec, self.opcode2idx, self.embedding_dim,
                                     self.hidden_size, self.rnn_layers,
                                     self.dropout_rate)

        self.rnn = nn.LSTM(self.hidden_size,
                           self.hidden_size,
                           num_layers=self.rnn_layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=self.dropout_rate)

        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, dfs_seq_list):
        # [batch, hidden_dim*2]

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

        return logit

    def attn_net(self, x, query, mask=None):  # soft attention（key=value=x）

        d_k = query.size(-1)  # d_k is the dim of query

        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # scores:[batch, seq_len, seq_len]

        p_attn = F.softmax(scores, dim=-1)

        context = torch.matmul(p_attn, x).sum(1)  # [batch, seq_len, hidden_dim*2]-> [batch, hidden_dim*2]

        return context, p_attn


class DFSAttention(nn.Module):
    def __init__(self, opid2vec, opcode2idx, embedding_dim, hidden_size, rnn_layers, dropout_rate):
        super(DFSAttention, self).__init__()

        self.opid2vec = opid2vec
        self.opcode2idx = opcode2idx

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.rnn_layers = rnn_layers
        self.dropout_rate = dropout_rate

        pretrained_dict = torch.from_numpy(self.opid2vec)

        self.op_emb = nn.Embedding(num_embeddings=len(self.opid2vec),
                                   embedding_dim=self.embedding_dim,
                                   dtype=torch.float32).from_pretrained(pretrained_dict).float()
        self.op_emb.weight.data[self.opcode2idx['<PAD>'], :] = 0.0   # set the weight of <PAD> to 0

        self.rnn = nn.LSTM(self.embedding_dim,
                           self.hidden_size,
                           num_layers=self.rnn_layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=self.dropout_rate)

        self.fc = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, inputs):

        inputs = self.op_emb(inputs)

        output, _ = self.rnn(inputs)

        query = self.dropout(output)
        attn_output, attn = self.attn_net(output, query)  #

        logit = self.fc(attn_output)   # [B * dim]

        return logit

    def attn_net(self, x, query, mask=None):  # soft attention（key=value=x）

        d_k = query.size(-1)  # d_k is the dim of query

        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # scores:[batch, seq_len, seq_len]

        p_attn = F.softmax(scores, dim=-1)

        context = torch.matmul(p_attn, x).sum(1)  # [batch, seq_len, hidden_dim*2]-> [batch, hidden_dim*2]

        return context, p_attn
