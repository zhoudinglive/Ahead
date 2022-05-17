#!/usr/bin/env python
# encoding: utf-8

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, device, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class SparseAttention(nn.Module):
    def __init__(self, input_size, hidden_size, device, context_size=0, dim_reduce=True):
        super(SparseAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.context_size = context_size
        self.dim_reduce = dim_reduce

        if dim_reduce:
            self.W = nn.Parameter(torch.zeros(size=(input_size, hidden_size)))
            nn.init.xavier_uniform_(self.W.data, gain=1.414)
        if context_size == 0:
            self.a = nn.Parameter(torch.zeros(size=(1, hidden_size * 2)))
        else:
            self.a = nn.Parameter(torch.zeros(size=(1, hidden_size * 2 + context_size)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x, adj, context=None):
        N = x.size(0)
        assert N == adj.size(0)

        edges = torch.nonzero(adj, as_tuple=False).t()
        if self.dim_reduce:
            x_hidden = torch.mm(x, self.W)
        else:
            x_hidden = x
        assert not torch.isnan(x_hidden).any()

        if self.context_size == 0:
            attn_concat = torch.cat([x_hidden[edges[0, :], :], x_hidden[edges[1, :], :]], dim=1).t()
        else:
            attn_concat = torch.cat([x_hidden[edges[0, :], :], x_hidden[edges[1, :], :], context], dim=1).t()
        attn_exp = torch.exp(-self.leakyrelu(self.a.mm(attn_concat).squeeze()))
        assert not torch.isnan(attn_exp).any()

        attn_exp = torch.sparse.FloatTensor(edges, attn_exp, torch.Size([N, N])).to_dense()
        attn_sum = torch.sum(attn_exp, dim=1).repeat(1, N)

        attn = torch.div(attn_exp, torch.reshape(attn_sum, (N, N)) + 1e-5)
        assert not torch.isnan(attn).any()

        return torch.mm(attn, x_hidden)


class HGCNLayer(nn.Module):
    def __init__(self, input_size, output_size, device, context_size=0):
        super(HGCNLayer, self).__init__()
        self.output_size = output_size
        self.context_size = context_size

        # type attention
        self.gcn_a = GraphConvolution(input_size, output_size, device)
        self.gcn_b = GraphConvolution(input_size, output_size, device)
        self.mlp_ab = nn.Linear(output_size * 2, output_size)
        self.attn_a = nn.Linear(input_size * 2, 1)
        self.attn_b = nn.Linear(input_size * 2, 1)

        # node attention
        self.node_attn_a = SparseAttention(input_size, output_size, device, context_size, dim_reduce=True)

        # fusion
        self.w_a = nn.Linear(output_size, output_size)
        self.w_b = nn.Linear(output_size, output_size)

    def forward(self, x, adj_a, adj_b, context=None):
        if self.context_size != 0:
            assert not context is None and context.size(1) == self.context_size

        N = x.size(0)
        type_sum_a, type_sum_b = torch.mm(adj_a, x), torch.mm(adj_b, x)
        attn_a = torch.sigmoid(self.attn_a(torch.cat([type_sum_a, x], dim=1)))
        attn_b = torch.sigmoid(self.attn_b(torch.cat([type_sum_b, x], dim=1)))

        # train with node attention
        x_a = self.node_attn_a(x, adj_a, context)
        x_b = self.gcn_b(x, adj_b)
        x = attn_a.repeat(1, self.output_size) * x_a + attn_b.repeat(1, self.output_size) * x_b
        x = torch.sigmoid(x)

        return x

    def prepare_attention_input(self, x):
        N = x.size(0)
        x_repeat_chunks = x.repeat_interleave(N, dim=0)
        x_repeat = x.expand(N, N, x.size(1)).reshape(-1, x.size(1))
        combine_matrix = torch.cat([x_repeat_chunks, x_repeat], dim=1)
        return combine_matrix.view(N, N, 2 * self.output_size)


class HGCN(nn.Module):
    def __init__(self, hiddens, dropout, device, context_size=0):
        super(HGCN, self).__init__()
        assert len(hiddens) > 0
        self.dropout = dropout
        self.hgcns = nn.ModuleList([HGCNLayer(hiddens[idx], hiddens[idx + 1], device, context_size) for idx in range(len(hiddens) - 1)])

    def forward(self, x, adj_a, adj_b, context=None):
        x = self.hgcns[0](x, adj_a, adj_b, context)
        for idx, layer in enumerate(self.hgcns[1:]):
            x = layer(x, adj_a, adj_b, context)
            if idx != len(self.hgcns) - 1:
                x = F.relu(x)
            x = F.dropout(x, self.dropout)
        return x
