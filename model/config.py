#!/usr/bin/env python
# encoding: utf-8

import torch


class Config(object):
    def __init__(self, args):
        self.DATA_PATH = args.data_path
        self.MODEL_SAVE_PATH = args.model_save_path
        self.COMPANY_SIZE = args.company_size
        self.TITLE_SIZE = args.title_size
        self.TIME_SIZE = args.time_size
        self.SENIOR_SIZE = args.senior_size
        self.PERSONAL_SIZE = args.personal_size
        self.MODEL_NAME = args.model_name

        self.GRU_INPUT_SIZE = args.gru_input_size
        self.GRU_HIDDEN_SIZE = args.gru_hidden_size
        self.GRU_HIDDEN_LAYER_SIZE = args.gru_hidden_layer_size

        self.GCN_HIDDENS = args.gcn_hiddens
        assert self.GCN_HIDDENS[-1] == self.GRU_INPUT_SIZE

        self.CONTEXT_CLASS = args.context_class
        self.CONTEXT_EMBED_SIZE = args.context_embed_size

        self.DEVICE = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
        self.EPOCH = args.epoch
        self.LEARN_RATE = args.learn_rate
        self.LEARN_RATE_DECAY = args.learn_rate_decay
        self.LEARN_RATE_DECAY_ROUND = args.learn_rate_decay_round
        self.BATCH_SIZE = args.batch_size
        self.WEIGHT_DECAY = args.weight_decay
        self.DROPOUT = args.dropout

        self.TRAIN_RATE = args.train_proportion
        self.VALID_RATE = args.valid_proportion
        self.TEST_RATE = 1 - self.TRAIN_RATE - self.VALID_RATE

