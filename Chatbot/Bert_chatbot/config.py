"""

@file  : config.py

@author: xiaolu

@time  : 2020-03-25

"""
import torch


class Config:
    # Gunakan GPU jika tersedia (ubah indeks cuda jika perlu, mis. cuda:0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dream_train_corpus_path = './data/chatbot.txt'
    bert_chinese_vocab = './data/vocab.txt'
    pretrain_model_path = './data/pytorch_model.bin'

    batch_size = 2

    learning_rate = 1e-5
    max_length = 100

    EPOCH = 1000




