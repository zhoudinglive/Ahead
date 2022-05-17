#!/usr/bin/env python
# encoding: utf-8

import torch
import numpy as np
from main import get_skill_embed


def evaluate_batch(prob_next_event, mask, target_event, only_last=False):
    correct_num_top1 = 0
    correct_num_top15 = 0
    correct_num_top30 = 0
    mmr = 0
    if only_last is False:
        predict_num = np.sum(mask)
        for prob_one_seq, mask_one_seq, target_one_seq in zip(
                prob_next_event, mask, target_event):
            for prob_one_event, target in zip(prob_one_seq[np.where(mask_one_seq == 1)],
                                              target_one_seq[np.where(mask_one_seq == 1)]):
                prob_sort = np.argsort(prob_one_event)
                event_top1 = np.argmax(prob_one_event)
                event_top15 = prob_sort[-15:]
                event_top30 = prob_sort[-30:]

                real_sort = prob_sort[::-1]
                if target in real_sort:
                    cur_index = np.where(real_sort==target)[0][0] + 1 # rank index from 1
                    mmr += 1.0/cur_index

                if target == event_top1:
                    correct_num_top1 += 1
                if target in event_top15:
                    correct_num_top15 += 1
                if target in event_top30:
                    correct_num_top30 += 1
    else:
        last_elem_idx = np.sum(mask, -1) - 1
        batch_size = mask.shape[0]
        prob_last = prob_next_event[np.arange(batch_size), last_elem_idx, :]
        target_last = target_event[np.arange(batch_size), last_elem_idx]
        predict_num = batch_size
        for prob_one_event, target in zip(prob_last, target_last):
            prob_sort = np.argsort(prob_one_event)
            event_top1 = np.argmax(prob_one_event)
            event_top15 = prob_sort[-15:]
            event_top30 = prob_sort[-30:]

            real_sort = prob_sort[::-1]
            if target in real_sort:
                cur_index = np.where(real_sort == target)[0][0] + 1  # rank index from 1
                mmr += 1.0 / cur_index

            if target == event_top1:
                correct_num_top1 += 1
            if target in event_top15:
                correct_num_top15 += 1
            if target in event_top30:
                correct_num_top30 += 1
    return correct_num_top1, correct_num_top15, correct_num_top30, predict_num, mmr


def evaluate(model, data, tag, config, internal_graph, external_graph, only_last=False):
    with torch.no_grad():
        company_num, title_num = 0, 0
        company_correct_top1, title_correct_top1 = 0, 0
        company_correct_top15, title_correct_top15 = 0, 0
        company_correct_top30, title_correct_top30 = 0, 0
        company_mmr, title_mmr = 0, 0

        duration_rmse, duration_mae, duration_num = 0., 0., 0.
        for seq_company, seq_mask_company, target_company, \
            seq_title, seq_mask_title, target_title, seq_time, pids in data.gen_batch(tag):

            seq_company, seq_mask_company, target_company = \
                data_process(seq_company, seq_mask_company, target_company, config.DEVICE)
            seq_title, seq_mask_title, target_title = \
                data_process(seq_title, seq_mask_title, target_title, config.DEVICE)
            first_time_input = np.full((seq_time.shape[0], 1), 24)
            seq_time_input = torch.LongTensor(seq_time[:, 1:]).to(config.DEVICE)
            batch_skill_embed = torch.Tensor(get_skill_embed(pids)).to(config.DEVICE)

            prob_company, prob_title, prob_duration, _, _, _, _ = model(seq_company, seq_title, seq_time_input, internal_graph, external_graph, batch_skill_embed)
            prob_company, prob_title, prob_duration = prob_company.cpu().numpy(), prob_title.cpu().numpy(), prob_duration.cpu().numpy()

            company_top1, company_top15, company_top30, cnt, mmr = evaluate_batch(
                prob_company, seq_mask_company.cpu().numpy(), target_company.cpu().numpy(), only_last)
            company_num += cnt
            company_correct_top1 += company_top1
            company_correct_top15 += company_top15
            company_correct_top30 += company_top30
            company_mmr += mmr

            title_top1, title_top15, title_top30, cnt, mmr = evaluate_batch(
                prob_title, seq_mask_title.cpu().numpy(), target_title.cpu().numpy(), only_last)
            title_num += cnt
            title_correct_top1 += title_top1
            title_correct_top15 += title_top15
            title_correct_top30 += title_top30
            title_mmr += mmr

            prob_duration, true_duration = prob_duration.flatten(), seq_time.flatten()
            for a, b in zip(prob_duration, true_duration):
                if b == 0:
                    continue
                duration_num += 1
                duration_mae += abs(a - b)
                duration_rmse += (a - b) ** 2

        cct1, cct15, cct30 = company_correct_top1 * 1.0 / company_num, \
                             company_correct_top15 * 1.0 / company_num, \
                             company_correct_top30 * 1.0 / company_num
        tct1, tct15, tct30 = title_correct_top1 * 1.0 / title_num, \
                             title_correct_top15 * 1.0 / title_num, \
                             title_correct_top30 * 1.0 / title_num
        duration_rmse /= duration_num
        duration_mae /= duration_num
        print("%s | Company: Acc@1=%.4f, Acc@15=%.4f, Acc@30=%.4f, MMR=%.4f | "
              "Title: Acc@1=%.4f, Acc@15=%.4f, Acc@30=%.4f, MMR=%.4f." % (
            tag, cct1, cct15, cct30, company_mmr / company_num,
            tct1, tct15, tct30, title_mmr / title_num
        ))
        print("%s | Duration: RMSE=%.4f, MAE=%.4f." % (tag, np.sqrt(duration_rmse), duration_mae))
        return cct1, tct1


def data_process(seq_event, seq_mask, target_event, device):
    seq_event = torch.LongTensor(seq_event).to(device)
    seq_mask = torch.LongTensor(seq_mask).to(device)
    target_event = torch.LongTensor(target_event).to(device)
    return seq_event, seq_mask, target_event


def normalize_graph(graph):
    rowsum = np.array(graph.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = np.diag(d_inv)
    norm_adj = d_mat.dot(graph).dot(d_mat)
    return norm_adj

