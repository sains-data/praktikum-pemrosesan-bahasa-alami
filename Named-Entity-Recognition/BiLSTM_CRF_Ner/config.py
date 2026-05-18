# -*- coding: utf-8 -*-
# @Time    : 2020/6/29 19:38
# @Author  : xiaolu
# @FileName: config.py
# @Software: PyCharm
import torch


class Config:
    device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')

    pickle_path = './data/renmindata.pkl'  # path file pkl hasil pra-pemrosesan

    load_model_path = None  # path checkpoint; None = latih dari awal

    batch_size = 128  # batch size
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    max_epoch = 20
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-5  # weight decay optimizer

    embedding_dim = 100
    hidden_dim = 200
    dropout = 0.2




