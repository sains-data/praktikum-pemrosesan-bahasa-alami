"""
Neural Probabilistic Language Model (NPLM) — prediksi kata dari 2 kata konteks.
Mata kuliah NLP Berbasis Transformer, Itera.
"""

import json
import os
import re

import jieba
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F


class NPLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NPLM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embedding = self.embed(inputs)
        batch_size = embedding.size(0)
        embedding = embedding.view(batch_size, -1)
        out = F.relu(self.linear1(embedding))
        out = self.linear2(out)
        return F.log_softmax(out, dim=-1)

    def extract(self, inputs):
        """Mengembalikan vektor embedding untuk indeks token."""
        return self.embed(inputs)


class DataTxt(Dataset):
    def __init__(self, data, vocab2id):
        self.data = data
        self.vocab2id = vocab2id

    def __getitem__(self, item):
        x_words, y_word = self.data[item][0], self.data[item][1]
        x = torch.LongTensor([self.vocab2id.get(x_words[0], 0), self.vocab2id.get(x_words[1], 0)])
        y = torch.LongTensor([self.vocab2id.get(y_word, 0)])
        return x, y

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    corpus_path = './data/corpus.txt'
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            'Buat berkas teks UTF-8 di Embedding/data/corpus.txt '
            '(novel atau artikel panjang).'
        )

    with open(corpus_path, 'r', encoding='utf-8') as f:
        data = f.read()

    # Tokenisasi (jieba untuk Mandarin; untuk Indonesia bisa diganti .split())
    temp = jieba.lcut(data)
    words = []
    for token in temp:
        token = token.strip()
        token = re.sub(r"[\s+\.\!\/_,$%^*(+\"']+|[+——！，。？、~@#￥%……&*（）]+", '', token)
        if len(token) != 0:
            words.append(token)

    trigrams = [([words[i], words[i + 1]], words[i + 2]) for i in range(len(words) - 2)]

    vocab = list(set(words))
    vocab2id = {'<UNK>': 0}
    for i, v in enumerate(vocab):
        vocab2id[v] = i + 1
    id2vocab = {i: v for v, i in vocab2id.items()}

    datatxt = DataTxt(trigrams, vocab2id)
    dataloader = DataLoader(datatxt, shuffle=True, batch_size=64)

    criterion = nn.NLLLoss()
    model = NPLM(len(vocab2id), embedding_dim=128, context_size=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch_count = 20
    losses = []
    step = 0

    for epoch in range(epoch_count):
        total_loss = 0.0
        for x, y in dataloader:
            step += 1
            y = torch.squeeze(y)
            optimizer.zero_grad()
            log_prob = model(x)
            loss = criterion(log_prob, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 500 == 0:
                print('Epoch: {}, langkah: {}, loss: {:.4f}'.format(epoch, step, loss.item()))

        avg = total_loss / len(dataloader)
        print('Epoch {} selesai — loss rata-rata: {:.4f}'.format(epoch, avg))
        losses.append(avg)

    with open('loss.json', 'w', encoding='utf-8') as f:
        json.dump(losses, f)

    sample_ids = torch.LongTensor([v for k, v in vocab2id.items() if k != '<UNK>'][:10])
    vec = model.extract(Variable(sample_ids))
    print('Contoh bentuk embedding:', vec.data.numpy().shape)
