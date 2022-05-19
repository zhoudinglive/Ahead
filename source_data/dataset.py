#!/usr/bin/env python
# encoding: utf-8

import sys
import json
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

np.set_printoptions(threshold=sys.maxsize)
COMPANY_COUNT = 1380 + 1
TITLE_COUNT = 2098 + 1


def load_json_file(f_path):
    with open(f_path, 'r', encoding='utf-8') as f_read:
        data = json.load(f_read)
    return data


class Data():
    def __init__(self, source_path, train_batch_size=32, valid_batch_size=128, train_proportion=0.8,
                 valid_proportion=0.1, test_proportion=0.1, seed=2020):
        data = load_json_file(source_path)

        pids = list() 
        companys = list()  
        titles = list() 
        com_pos = list() 
        durations = list()  
        start_times = list()  

        for k in data:
            pids.append(k)
            companys.append(data[k]['company'])
            titles.append(data[k]['title'])
            com_pos.append(data[k]['company_with_title'])
            start_time = data[k]['start_time']

            start_times.append(start_time)
            duration = list()
            for i, j in zip(start_time[:-1], start_time[1:]):
                duration.append(j - i)
            durations.append(duration)

        pids = np.array(pids)
        companys = np.array(companys)
        titles = np.array(titles)
        com_pos = np.array(com_pos)
        durations = np.array(durations)
        print(companys.shape)
        print(durations.shape)

        for idx in range(len(companys)):
            for jdx in range(len(companys[idx])):
                companys[idx][jdx] += 1

        for idx in range(len(titles)):
            for jdx in range(len(titles[idx])):
                titles[idx][jdx] += 1

        for idx in range(len(durations)):
            for jdx in range(len(durations[idx])):
                durations[idx][jdx] += 1

        if seed is None:
            randnum = random.randint(0, 100)
        else:
            randnum = seed
        random.seed(randnum)
        random.shuffle(companys)
        random.seed(randnum)
        random.shuffle(titles)
        random.seed(randnum)
        random.shuffle(com_pos)
        random.seed(randnum)
        random.shuffle(durations)

        self.size = dict()
        self.size['total_sample'] = len(companys)
        print('the total sample is:{}'.format(self.size['total_sample']))
        self.size['train'] = int(np.ceil(self.size['total_sample'] * train_proportion))
        self.size['valid'] = int(np.ceil(self.size['total_sample'] * valid_proportion))
        self.size['test'] = self.size['total_sample'] - self.size['train'] - self.size['valid']
        self.batch_ind = dict()
        self.batch_ind['train'] = np.arange(int(np.ceil(self.size['train'] / train_batch_size)))
        self.batch_ind['valid'] = np.arange(int(np.ceil(self.size['valid'] / valid_batch_size)))
        self.batch_ind['test'] = np.arange(int(np.ceil(self.size['test'] / valid_batch_size)))
        self.split_company = dict()
        self.split_title = dict()
        self.split_com_pos = dict()
        self.split_duration = dict()
        self.split_pids = dict()

        self.split_pids['train'] = pids[:self.size['train']]
        self.split_pids['valid'] = pids[self.size['train']:(self.size['train'] + self.size['valid'])]
        self.split_pids['test'] = pids[(self.size['train'] + self.size['valid']):]

        self.split_company['train'] = companys[:self.size['train']]
        self.split_company['valid'] = companys[self.size['train']:(self.size['train'] + self.size['valid'])]
        self.split_company['test'] = companys[(self.size['train'] + self.size['valid']):]

        self.split_title['train'] = titles[:self.size['train']]
        self.split_title['valid'] = titles[self.size['train']:(self.size['train'] + self.size['valid'])]
        self.split_title['test'] = titles[(self.size['train'] + self.size['valid']):]

        self.split_com_pos['train'] = com_pos[:self.size['train']]
        self.split_com_pos['valid'] = com_pos[self.size['train']:(self.size['train'] + self.size['valid'])]
        self.split_com_pos['test'] = com_pos[(self.size['train'] + self.size['valid']):]

        self.split_duration['train'] = durations[:self.size['train']]
        self.split_duration['valid'] = durations[self.size['train']:(self.size['train'] + self.size['valid'])]
        self.split_duration['test'] = durations[(self.size['train'] + self.size['valid']):]

        print('the total train batch is:{}'.format(len(self.batch_ind['train'])))
        print('the total valid batch is:{}'.format(len(self.batch_ind['valid'])))
        print('the total test batch is:{}'.format(len(self.batch_ind['test'])))

        self.batches = {'train': {'pids':{}, 'train_company': {}, 'train_title': {}, 'train_com_pos': {}, 'target_company': {},
                                  'target_title': {}, 'target_com_pos': {}, 'mask': {}, 'duration': {}},
                        'valid': {'pids':{}, 'train_company': {}, 'train_title': {}, 'train_com_pos': {}, 'target_company': {},
                                  'target_title': {}, 'target_com_pos': {}, 'mask': {}, 'duration': {}},
                        'test': {'pids':{}, 'train_company': {}, 'train_title': {}, 'train_com_pos': {}, 'target_company': {},
                                 'target_title': {}, 'target_com_pos': {}, 'mask': {}, 'duration': {}}}

        for tag in ['train', 'valid', 'test']:
            if tag == 'train':
                offset = 0
                cap = self.size['train']
                batch_size = train_batch_size
            elif tag == 'valid':
                offset = self.size['train']
                cap = self.size['train'] + self.size['valid']
                batch_size = valid_batch_size
            else:
                offset = self.size['train'] + self.size['valid']
                cap = self.size['total_sample']
                batch_size = valid_batch_size
            for batch_index in self.batch_ind[tag]:
                i, j = batch_index * batch_size + offset, min((batch_index + 1) * batch_size + offset, cap)
                max_length = max([len(seq) for seq in companys[i:j]]) - 1 
                self.batches[tag]['train_company'][batch_index] = np.array(
                    [seq[:-1] + (max_length - len(seq[:-1])) * [0] for seq in companys[i:j]])
                self.batches[tag]['train_title'][batch_index] = np.array(
                    [seq[:-1] + (max_length - len(seq[:-1])) * [0] for seq in titles[i:j]])
                self.batches[tag]['train_com_pos'][batch_index] = np.array(
                    [seq[:-1] + (max_length - len(seq[:-1])) * [0] for seq in com_pos[i:j]])
                self.batches[tag]['duration'][batch_index] = np.array(
                    [seq[:] + (max_length - len(seq[:])) * [0] for seq in durations[i:j]])
                self.batches[tag]['pids'][batch_index] = np.array(pids[i:j]).astype(int)

                self.batches[tag]['mask'][batch_index] = np.array(
                    [len(seq[:-1]) * [1] + (max_length - len(seq[:-1])) * [0] for seq in companys[i:j]])
                self.batches[tag]['target_company'][batch_index] = np.array(
                    [seq[1:] + (max_length - len(seq[1:])) * [0] for seq in companys[i:j]])
                self.batches[tag]['target_title'][batch_index] = np.array(
                    [seq[1:] + (max_length - len(seq[1:])) * [0] for seq in titles[i:j]])
                self.batches[tag]['target_com_pos'][batch_index] = np.array(
                    [seq[1:] + (max_length - len(seq[1:])) * [0] for seq in com_pos[i:j]])

    def gen_batch(self, tag):
        for ind in np.random.permutation(self.batch_ind[tag]):
            pids = self.batches[tag]['pids'][ind]
            seq_company = self.batches[tag]['train_company'][ind]
            seq_mask_company = self.batches[tag]['mask'][ind]
            target_company = self.batches[tag]['target_company'][ind]

            seq_title = self.batches[tag]['train_title'][ind]
            seq_mask_title = self.batches[tag]['mask'][ind]
            target_title = self.batches[tag]['target_title'][ind]

            seq_time = self.batches[tag]['duration'][ind]

            yield seq_company, seq_mask_company, target_company, seq_title, seq_mask_title, target_title, seq_time, pids


def split_generator(num_category, context_adj):
    assert type(context_adj) == np.ndarray

    data = sorted(context_adj[context_adj.nonzero()])
    piece = int(len(data) / num_category)
    pieces = [data[piece * idx] for idx in range(num_category)]
    return pieces + [1.]


def context_index(num_category, context_adj, scale=True):
    def get_idx(value):
        left, right = 0, len(intervals)
        while left <= right:
            mid = (left + right) >> 1
            if (mid - left) == 1 and intervals[mid] >= value >= intervals[left]:
                return left
            elif (right - mid) == 1 and intervals[right] >= value >= intervals[mid]:
                return mid
            elif value > intervals[mid]:
                left = mid
            else:
                right = mid

    def row_normal(adj):
        row_sum = np.expand_dims(np.sum(adj, axis=1), axis=1).repeat(adj.shape[1], axis=1)
        normal_adj = adj / row_sum
        normal_adj[np.isnan(normal_adj)] = 0.
        return normal_adj

    if scale is True:
        context_adj = row_normal(context_adj)
    intervals = split_generator(num_category, context_adj)

    for idx in range(context_adj.shape[0]):
        for jdx in range(context_adj.shape[1]):
            if context_adj[idx][jdx] != 0:
                context_adj[idx][jdx] = get_idx(context_adj[idx][jdx])
    return context_adj


def context_graph(num_category):
    dur_company_context = context_index(num_category, np.load("source_data/dur_of_company.npy"))
    dur_title_context = context_index(num_category, np.load("source_data/dur_of_title.npy"))
    dur_context_graph = np.zeros((COMPANY_COUNT + TITLE_COUNT, COMPANY_COUNT + TITLE_COUNT))
    dur_context_graph[:COMPANY_COUNT, :COMPANY_COUNT] = dur_company_context
    dur_context_graph[COMPANY_COUNT:, COMPANY_COUNT:] = dur_title_context

    ppr_company_context = context_index(num_category, np.load("source_data/company_ppr.npy"), scale=False)
    ppr_title_context = context_index(num_category, np.load("source_data/title_ppr.npy"), scale=False)
    ppr_context_graph = np.zeros((COMPANY_COUNT + TITLE_COUNT, COMPANY_COUNT + TITLE_COUNT))
    ppr_context_graph[:COMPANY_COUNT, :COMPANY_COUNT] = ppr_company_context
    ppr_context_graph[COMPANY_COUNT:, COMPANY_COUNT:] = ppr_title_context
    return dur_context_graph, ppr_context_graph