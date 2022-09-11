#!/usr/bin/env python
# encoding: utf-8


import argparse
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import utils as nn_utils
from torch.nn.parameter import Parameter

import json
import time
from datetime import datetime
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from source_data.dataset import Data
from model.utils import *
from model.ahead import HierarchicalCareer
from model.config import Config

skill_embed = np.load("source_data/mock_skill_embed_small.npy", allow_pickle=True).tolist()
pid_str_dict = None
with open("source_data/mock_pid_str_dict.json", "r") as f:
    pid_str_dict = json.load(f)


def adjust_learning_rate(optimizer, decay_round, decay_rate, epoch, lr):
    assert 0 < decay_rate < 1
    lr = lr * (decay_rate ** (epoch // decay_round))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_skill_embed(pids):
    pids_embed = list()
    tmp = np.zeros((300, ))
    for pid in pids:
        pid_str = pid_str_dict.get(pid)
        current_embed = skill_embed.get(pid_str)
        if current_embed is not None:
            pids_embed.append(current_embed)
        else:
            pids_embed.append(tmp)
    pids_embed = np.array(pids_embed)
    return pids_embed


def train(config):
    model = HierarchicalCareer(config)
    if torch.cuda.is_available():
        model = model.to(config.DEVICE)

    loss_company = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss_title = torch.nn.CrossEntropyLoss(ignore_index=0)
    loss_duration = torch.nn.MSELoss()
    loss_company_graph = torch.nn.MSELoss()
    loss_title_graph = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARN_RATE, weight_decay=config.WEIGHT_DECAY)

    internal_graph = torch.FloatTensor(
        np.load("source_data/mock_internal_graph.npy")).to(config.DEVICE)
    external_graph = torch.FloatTensor(
        np.load("source_data/mock_external_graph.npy")).to(config.DEVICE)

    company_graph_true = internal_graph[:1381, :1381]
    title_graph_true = internal_graph[1381:, 1381:]

    max_epoch, max_company_acc1, max_title_acc1 = 0, -1., -1.
    for epoch in range(config.EPOCH):
        epoch_loss, epoch_loss_cg, epoch_loss_tg = list(), list(), list()
        internal_graph_mask = F.dropout(internal_graph, p=0.0)
        external_graph_mask = F.dropout(external_graph, p=0.0)

        start_time = time.time()
        for seq_company, seq_mask_company, target_company, \
            seq_title, seq_mask_title, target_title, seq_time, pids in data.gen_batch('train'):

            seq_company, seq_mask_company, target_company = \
                data_process(seq_company, seq_mask_company, target_company, config.DEVICE)
            seq_title, seq_mask_title, target_title = \
                data_process(seq_title, seq_mask_title, target_title, config.DEVICE)

            seq_time_input = torch.LongTensor(seq_time[:, 1:]).to(config.DEVICE)
            batch_skill_embed = torch.Tensor(get_skill_embed(pids)).to(config.DEVICE)

            company_pred, title_pred, duration_pred, _, _, company_graph, title_graph = model(
                seq_company,
                seq_title,
                seq_time_input,
                internal_graph_mask,
                external_graph_mask,
                batch_skill_embed)
            company_pred, title_pred = company_pred.view(-1, config.COMPANY_SIZE), title_pred.view(-1, config.TITLE_SIZE)
            company_true, title_true = target_company.view(-1), target_title.view(-1)
            duration_pred, duration_true = duration_pred.squeeze().view(-1), torch.Tensor(seq_time).to(config.DEVICE).view(-1)

            loss = loss_company(company_pred, company_true) \
                   + loss_title(title_pred, title_true) \
                   + loss_duration(duration_pred, duration_true)
            loss_company_g = 0.1 * loss_company_graph(company_graph.reshape(-1, 1), company_graph_true.reshape(-1, 1))
            loss_title_g = 0.1 * loss_title_graph(title_graph.reshape(-1, 1), title_graph_true.reshape(-1, 1))

            loss += loss_company_g + loss_title_g

            epoch_loss.append(loss.item())
            epoch_loss_cg.append(loss_company_g.item())
            epoch_loss_tg.append(loss_title_g.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(torch.sum(company_graph[547]))
        print("Epoch-%d, Train Loss=%.4f, CG Loss=%.4f, TG Loss=%.4f, Train Latency=%.2fs."
              % (epoch, np.mean(epoch_loss), np.mean(epoch_loss_cg), np.mean(epoch_loss_tg), time.time() - start_time))
        with torch.no_grad():
            cct1, tct1 = evaluate(model, data, "valid", config, internal_graph, external_graph, only_last=False)
            if cct1 + tct1 > max_company_acc1 + max_title_acc1:
                max_epoch, max_company_acc1, max_title_acc1 = epoch, cct1, tct1
                torch.save(model, config.MODEL_SAVE_PATH + config.MODEL_NAME)
    print("Best Epoch is %d." % max_epoch)


def test(config):
    model = torch.load(config.MODEL_SAVE_PATH + config.MODEL_NAME, map_location=config.DEVICE)
    internal_graph = torch.FloatTensor(
        np.load("source_data/mock_internal_graph.npy")).to(config.DEVICE)
    external_graph = torch.FloatTensor(
        np.load("source_data/mock_external_graph.npy")).to(config.DEVICE)
    with torch.no_grad():
        cct1, tct1 = evaluate(model, data, "test", config, internal_graph, external_graph, only_last=False)


data = None
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of Career Path Learning.")
    parser.add_argument('--data_path', default='source_data/mock_final_records.json')
    parser.add_argument('--model_save_path', default='model_save/')
    parser.add_argument('--company_size', default=1380 + 1, type=int)
    parser.add_argument('--title_size', default=2098 + 1, type=int)
    parser.add_argument('--time_size', default=23 + 2, type=int)
    parser.add_argument('--senior_size', default= 23 * 4 + 1, type=int)
    parser.add_argument('--personal_size', default=459309, type=int)
    parser.add_argument('--model_name', default="ahead_duration", type=str)

    parser.add_argument('--gru_input_size', default=128, type=int)
    parser.add_argument('--gru_hidden_size', default=256, type=int)
    parser.add_argument('--gru_hidden_layer_size', default=1, type=int)

    parser.add_argument('--gcn_hiddens', default=[256, 128], type=int, nargs='+')

    parser.add_argument('--context_class', default=20, type=int)
    parser.add_argument('--context_embed_size', default=128, type=int)

    parser.add_argument('--device', default="0", type=str)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--learn_rate', default=0.0005, type=float)
    parser.add_argument('--learn_rate_decay', default=0.5, type=float)
    parser.add_argument('--learn_rate_decay_round', default=10, type=int)
    parser.add_argument('--batch_size', default=4096, type=int)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--dropout', default=0.15, type=float)

    parser.add_argument('--train_proportion', default=0.8, type=float)
    parser.add_argument('--valid_proportion', default=0.1, type=float)
    args = parser.parse_args()

    config = Config(args)
    data = Data(
        source_path=config.DATA_PATH,
        train_batch_size=config.BATCH_SIZE,
        valid_batch_size=config.BATCH_SIZE,
        train_proportion=config.TRAIN_RATE,
        valid_proportion=config.VALID_RATE,
        test_proportion=config.TEST_RATE
    )

    train(config)
    test(config)