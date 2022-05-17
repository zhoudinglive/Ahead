#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
from torch.autograd import Variable
import math


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.x2h = nn.Linear(input_size, hidden_size * 3)
        self.h2h = nn.Linear(hidden_size, hidden_size * 3)
        self.mlp = nn.Linear(input_size + hidden_size, 1)
        self.reset_parameters()

    def forward(self, x, a_pre, b_pre):
        x = x.view(-1, x.size(1))
        x_copy = x.unsqueeze(dim=1).expand(x.size(0), 2, x.size(1))
        previous = torch.cat([a_pre.unsqueeze(dim=1), b_pre.unsqueeze(dim=1)], dim=1)
        attention_coefficient = torch.softmax(self.mlp(torch.cat([x_copy, previous], dim=2)).squeeze(), dim=1)

        attention_coefficient_ = attention_coefficient.unsqueeze(dim=2)\
            .expand(attention_coefficient.size(0), attention_coefficient.size(1), self.hidden_size)
        h_pre = torch.sum(attention_coefficient_ * previous, dim=1)

        gate_x = self.x2h(x).squeeze()
        gate_h = self.h2h(h_pre).squeeze()

        x_r, x_u, x_c = gate_x.chunk(3, 1)
        h_r, h_u, h_c = gate_h.chunk(3, 1)

        r_t = torch.sigmoid(x_r + h_r)
        u_t = torch.sigmoid(x_u + h_u)
        c_t = torch.tanh(x_c + (r_t * h_c))
        h_t = c_t + u_t * (h_pre - c_t)

        return h_t, attention_coefficient.unsqueeze(dim=1)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class CareerGRU(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layer, company_size, title_size, device, use_mlp=True):
        super(CareerGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer
        self.device = device
        self.use_mlp = use_mlp

        self.company_cell = GRUCell(input_size, hidden_size)
        self.title_cell = GRUCell(input_size, hidden_size)

        self.company_mlp = nn.Linear(hidden_size, company_size)
        self.title_mlp = nn.Linear(hidden_size, title_size)

    def forward(self, company_x, title_x):
        batch_size, seq_len, embed_size = company_x.size()
        company_h = Variable(
            torch.zeros(self.hidden_layer, batch_size, self.hidden_size)).to(self.device)[0, :, :]
        title_h = Variable(
            torch.zeros(self.hidden_layer, batch_size, self.hidden_size)).to(self.device)[0, :, :]

        company_out = torch.zeros(batch_size, seq_len, self.hidden_size).to(self.device)
        title_out = torch.zeros(batch_size, seq_len, self.hidden_size).to(self.device)

        company_attn, title_attn = list(), list()
        for idx in range(seq_len):
            
            new_company_h, cattn = self.company_cell(company_x[:, idx, :], company_h, title_h)
            new_title_h, tattn = self.title_cell(title_x[:, idx, :], company_h, title_h)
            company_h, title_h = new_company_h, new_title_h

            company_out[:, idx, :] = company_h
            title_out[:, idx, :] = title_h
            
            company_attn.append(cattn)
            title_attn.append(tattn)

        if self.use_mlp:
            company_out = self.company_mlp(company_out)
            title_out = self.title_mlp(title_out)
        
        return company_out, title_out, torch.cat(company_attn, dim=1), torch.cat(title_attn, dim=1)
