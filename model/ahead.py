#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from model.dgru import CareerGRU
from model.hgcn import HGCN
from model.utils import *
from source_data.dataset import context_graph


class HierarchicalCareer(nn.Module):
    def __init__(self, config):
        super(HierarchicalCareer, self).__init__()
        self.config = config

        dur_context_graph, ppr_context_graph = context_graph(config.CONTEXT_CLASS)
        self.dur_context_graph = torch.LongTensor(dur_context_graph).to(config.DEVICE)
        self.ppr_context_graph = torch.LongTensor(ppr_context_graph).to(config.DEVICE)

        # indexes
        self.company_idx = torch.LongTensor(range(config.COMPANY_SIZE)).to(config.DEVICE)
        self.title_idx = torch.LongTensor(range(config.TITLE_SIZE)).to(config.DEVICE)
        self.time_idx = torch.LongTensor(range(config.TIME_SIZE)).to(config.DEVICE)
        self.company_title_idx = torch.LongTensor(range(config.COMPANY_SIZE + config.TITLE_SIZE)).to(config.DEVICE)

        self.context_embed = nn.Embedding(config.CONTEXT_CLASS, config.CONTEXT_EMBED_SIZE)
        self.dur_context_embed = nn.Embedding(config.CONTEXT_CLASS, config.CONTEXT_EMBED_SIZE)
        self.ppr_context_embed = nn.Embedding(config.CONTEXT_CLASS, config.CONTEXT_EMBED_SIZE)
        self.company_title_embed = nn.Embedding(config.COMPANY_SIZE + config.TITLE_SIZE, config.GCN_HIDDENS[0])
        self.hgcn = HGCN(config.GCN_HIDDENS, config.DROPOUT, config.DEVICE, context_size=config.CONTEXT_EMBED_SIZE)
        self.gru = CareerGRU(config.GRU_INPUT_SIZE * 3, config.GRU_HIDDEN_SIZE, config.GRU_HIDDEN_LAYER_SIZE,
                             config.COMPANY_SIZE, config.TITLE_SIZE, config.DEVICE, use_mlp=False)
        self.time_embed = nn.Embedding(config.TIME_SIZE, config.GRU_INPUT_SIZE)
        self.senior_embed = nn.Embedding(config.SENIOR_SIZE, config.GRU_INPUT_SIZE)

        self.skill_mlp = nn.Linear(300, config.GRU_INPUT_SIZE)

        self.company_readout = nn.Linear(config.GRU_HIDDEN_SIZE, config.COMPANY_SIZE)
        self.title_readout = nn.Linear(config.GRU_HIDDEN_SIZE, config.TITLE_SIZE)
        self.duration_readout = nn.Sequential(
            nn.Linear(config.GRU_HIDDEN_SIZE * 2, config.GRU_HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(config.GRU_HIDDEN_SIZE, 1),
            nn.ReLU()
        )
        self.dim_reduce_company = nn.Linear(config.GRU_INPUT_SIZE * 3, config.GRU_INPUT_SIZE)
        self.transform_title = nn.Sequential(
            nn.Linear(config.GRU_INPUT_SIZE * 3, config.GRU_INPUT_SIZE * 3),
            nn.ReLU()
        )

        self.company_graph_rebuild = nn.Sequential(
            nn.Linear(config.COMPANY_SIZE, config.COMPANY_SIZE),
            nn.Sigmoid()
        )
        self.title_graph_rebuild = nn.Sequential(
            nn.Linear(config.TITLE_SIZE, config.TITLE_SIZE),
            nn.Sigmoid()
        )


    def forward(self, seq_company, seq_title, seq_time, internal_graph, external_graph, batch_skill_embed=None):
        dur_context_idx = self.dur_context_graph[torch.nonzero(internal_graph, as_tuple=True)]
        ppr_context_idx = self.ppr_context_graph[torch.nonzero(internal_graph, as_tuple=True)]

        company_title_embed = self.company_title_embed(self.company_title_idx)
        dur_context_embed = self.dur_context_embed(dur_context_idx)
        ppr_context_embed = self.ppr_context_embed(ppr_context_idx)
        company_title_embed_ = self.hgcn(company_title_embed, internal_graph, external_graph,
                                        dur_context_embed + ppr_context_embed)

        company_embed = company_title_embed_[0:self.config.COMPANY_SIZE]
        title_embed = company_title_embed_[self.config.COMPANY_SIZE:]
        company_gru_input = company_embed[seq_company]
        title_gru_input = title_embed[seq_title]

        first_time = torch.zeros((seq_time.size(0), 1, self.config.GRU_INPUT_SIZE), device=self.config.DEVICE, requires_grad=True)
        time_embed = self.time_embed(seq_time)
        time_embed = torch.cat([first_time, time_embed], dim=1)

        batch_skill_embed = self.skill_mlp(batch_skill_embed).unsqueeze(dim=1).expand(-1, time_embed.size(1), -1)

        company_gru_input = torch.cat([company_gru_input, time_embed, batch_skill_embed], dim=2)
        title_gru_input = torch.cat([title_gru_input, time_embed, batch_skill_embed], dim=2)

        company_gru_hidden, title_gru_hidden, company_attn, title_attn = self.gru(company_gru_input, title_gru_input)
        company_gru_output = self.company_readout(company_gru_hidden)
        title_gru_output = self.title_readout(title_gru_hidden)
        duration_output = self.duration_readout(torch.cat([company_gru_hidden, title_gru_hidden], dim=2))

        company_embed_ = company_title_embed[0:self.config.COMPANY_SIZE]
        title_embed_ = company_title_embed[self.config.COMPANY_SIZE:]
        company_graph = self.company_graph_rebuild(torch.mm(company_embed_, torch.transpose(company_embed_, 0, 1)))
        title_graph = self.title_graph_rebuild(torch.mm(title_embed_, torch.transpose(title_embed_, 0, 1)))
        return company_gru_output, title_gru_output, duration_output, company_attn, title_attn, company_graph, title_graph