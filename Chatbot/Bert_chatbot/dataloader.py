"""

@file  : dataloader.py

@author: xiaolu

@time  : 2020-03-25

"""
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
from tokenizer import load_bert_vocab, Tokenizer
from config import Config


def read_corpus(data_path):
    """Membaca korpus dialog: satu baris per pasangan pertanyaan=jawaban."""
    sents_src = []
    sents_tgt = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            q, a = line.split('=')
            sents_src.append(q)
            sents_tgt.append(a)
    return sents_src, sents_tgt


class DreamDataset(Dataset):
    """Dataset dialog: pasangan sumber dan target dari file teks."""

    def __init__(self):
        super(DreamDataset, self).__init__()
        # Muat seluruh korpus ke memori
        self.sents_src, self.sents_tgt = read_corpus(Config.dream_train_corpus_path)
        self.word2idx = load_bert_vocab()
        self.idx2word = {k: v for v, k in self.word2idx.items()}
        self.tokenizer = Tokenizer(self.word2idx)

    def __getitem__(self, i):
        # Ambil satu sampel
        src = self.sents_src[i]
        tgt = self.sents_tgt[i]

        token_ids, token_type_ids = self.tokenizer.encode(src, tgt)
        output = {
            "token_ids": token_ids,
            "token_type_ids": token_type_ids,
        }
        return output

    def __len__(self):
        return len(self.sents_src)


def collate_fn(batch):
    """Padding dinamis per batch."""

    def padding(indice, max_length, pad_idx=0):
        """Padding token; token_type_id bagian jawaban diisi 1 (bukan 0)."""
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    token_ids = [data["token_ids"] for data in batch]
    max_length = max([len(t) for t in token_ids])
    token_type_ids = [data["token_type_ids"] for data in batch]

    token_ids_padded = padding(token_ids, max_length)
    token_type_ids_padded = padding(token_type_ids, max_length)
    target_ids_padded = token_ids_padded[:, 1:].contiguous()

    return token_ids_padded, token_type_ids_padded, target_ids_padded

