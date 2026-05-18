"""
Implementasi edukatif ALBERT: embedding faktorisasi + berbagi bobot antar-lapis.
Mata kuliah NLP Berbasis Transformer, Itera.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from typing import NamedTuple
import torch.nn.functional as F
import re
from random import *
from torch import optim


class Config:
    """Hyperparameter model ALBERT mini."""
    def __init__(self):
        self.embedding = 24
        self.hidden = 64
        self.hidden_ff = 256
        self.n_layers = 12
        self.n_heads = 8
        self.max_len = 30
        self.n_segments = 2
        self.vocab_size = 29


def gelu(x):
    """Fungsi aktivasi GELU."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def split_last(x, shape):
    '''
    split the last dimension to given shape
    :param x:
    :param shape:
    :return:
    '''
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


def merge_last(x, n_dims):
    """Gabungkan dimensi terakhir."""
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)


class LayerNorm(nn.Module):
    """Layer normalization."""
    def __init__(self, cfg, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(cfg.hidden))
        self.beta = nn.Parameter(torch.zeros(cfg.hidden))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    """Embedding faktorisasi: vocab→E→H, ditambah posisi dan segmen."""
    def __init__(self, cfg):
        super().__init__()
        # Original BERT Embedding    word_embedding == hidden_size
        # self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden)  # token embedding

        self.tok_embed1 = nn.Embedding(cfg.vocab_size, cfg.embedding)
        self.tok_embed2 = nn.Linear(cfg.embedding, cfg.hidden)

        self.pos_embed = nn.Embedding(cfg.max_len, cfg.hidden)   # position embedding
        self.seg_embed = nn.Embedding(cfg.n_segments, cfg.hidden)   # segment(token type) embedding

        self.norm = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (S,) -> (B, S)

        # factorized embedding
        e = self.tok_embed1(x)
        e = self.tok_embed2(e)
        e = e + self.pos_embed(pos) + self.seg_embed(seg)
        # return self.drop(self.norm(e))
        return self.norm(e)


class MultiHeadedSelfAttention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_k = nn.Linear(cfg.hidden, cfg.hidden)
        self.proj_v = nn.Linear(cfg.hidden, cfg.hidden)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None  # for visualization
        self.n_heads = cfg.n_heads

    def forward(self, x, mask):
        """Self-attention: Q, K, V dari x; bentuk (B, H, S, W)."""
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)

        q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])

        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        # if mask is not None:
        #     mask = mask[:, None, None, :].float()
        #     scores -= 10000.0 * (1.0 - mask)

        # scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)

        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()

        h = merge_last(h, 2)

        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """Feed-forward per posisi."""
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden, cfg.hidden_ff)
        self.fc2 = nn.Linear(cfg.hidden_ff, cfg.hidden)
        # self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu(self.fc1(x)))


class Transformer(nn.Module):
    '''
    Transformer with Self-Attentive Blocks
    '''
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

        # Berbagi bobot: modul attention & FFN yang sama diulang n_layers kali
        self.n_layers = cfg.n_layers
        self.attn = MultiHeadedSelfAttention(cfg)

        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.norm1 = LayerNorm(cfg)

        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        # self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for _ in range(self.n_layers):
            # h = block(h, mask)
            h = self.attn(h, mask)
            h = self.norm1(h + self.proj(h))
            h = self.norm2(h + self.pwff(h))
        return h


class BertModel4Pretrain(nn.Module):
    "Bert Model for Pretrain : Masked LM and next sentence classification"
    def __init__(self, cfg):
        super().__init__()
        self.transformer = Transformer(cfg)
        self.fc = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ1 = nn.Tanh()
        self.linear = nn.Linear(cfg.hidden, cfg.hidden)
        self.activ2 = gelu
        self.norm = LayerNorm(cfg)
        self.classifier = nn.Linear(cfg.hidden, 2)

        # decoder is shared with embedding layer
        # project hidden layer to embedding layer
        embed_weight2 = self.transformer.embed.tok_embed2.weight
        n_hidden, n_embedding = embed_weight2.size()
        self.decoder1 = nn.Linear(n_hidden, n_embedding, bias=False)
        self.decoder1.weight.data = embed_weight2.data.t()

        # project embedding layer to vocabulary layer
        embed_weight1 = self.transformer.embed.tok_embed1.weight
        n_vocab, n_embedding = embed_weight1.size()
        self.decoder2 = nn.Linear(n_embedding, n_vocab, bias=False)
        self.decoder2.weight = embed_weight1

        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, input_mask, masked_pos):
        h = self.transformer(input_ids, segment_ids, input_mask)
        pooled_h = self.activ1(self.fc(h[:, 0]))
        masked_pos = masked_pos[:, :, None].expand(-1, -1, h.size(-1))
        h_masked = torch.gather(h, 1, masked_pos)
        h_masked = self.norm(self.activ2(self.linear(h_masked)))

        logits_lm = self.decoder2(self.decoder1(h_masked)) + self.decoder_bias
        logits_clsf = self.classifier(pooled_h)

        return logits_lm, logits_clsf


def make_batch():
    """Batch contoh format BERT (MLM + NSP)."""
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))  # sample random index in sentences
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [vocab2id['[CLS]']] + tokens_a + [vocab2id['[SEP]']] + tokens_b + [vocab2id['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        n_pred = min(max_pred, max(1, int(round(len(input_ids) * 0.15))))

        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != vocab2id['[CLS]'] and token != vocab2id['[SEP]']]
        shuffle(cand_maked_pos)

        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:
                input_ids[pos] = vocab2id['[MASK]']
            elif random() < 0.5:
                index = randint(0, vocab_size - 1)
                input_ids[pos] = vocab2id[id2vocab[index]]

        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)

        if tokens_a_index + 1 == tokens_b_index and positive < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size/2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            negative += 1
    return batch


if __name__ == '__main__':
    cfg = Config()

    maxlen = 30
    batch_size = 6
    max_pred = 5

    text = (
        'Hello, how are you? I am Romeo.\n'
        'Hello, Romeo My name is Juliet. Nice to meet you.\n'
        'Nice meet you too. How are you today?\n'
        'Great. My baseball team won the competition.\n'
        'Oh Congratulations, Juliet\n'
        'Thanks you Romeo'
    )

    sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')
    word_list = list(set(" ".join(sentences).split()))
    vocab2id = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(word_list):
        vocab2id[w] = i + 4
    id2vocab = {i: w for i, w in enumerate(vocab2id)}
    vocab_size = len(vocab2id)
    # print(vocab_size)

    token_list = list()
    for sentence in sentences:
        arr = [vocab2id[s] for s in sentence.split()]
        token_list.append(arr)

    model = BertModel4Pretrain(cfg)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    batch = make_batch()
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
    # print(masked_tokens)
    # # [4, 23, 9, 0, 0], [7, 15, 0, 0, 0], [20, 0, 0, 0, 0], [23, 10, 0, 0, 0], [8, 21, 22, 0, 0], [9, 6, 20, 0, 0]
    # print(masked_pos)
    # # [10, 13, 1, 0, 0], [1, 7, 0, 0, 0], [6, 0, 0, 0, 0], [4, 13, 0, 0, 0], [8, 11, 5, 0, 0], [7, 6, 10, 0, 0]
    # print(isNext)
    # # (False, True, False, False, True, True)
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
        torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos), torch.LongTensor(isNext)

    for epoch in range(100):
        optimizer.zero_grad()
        # forward(self, input_ids, segment_ids, input_mask, masked_pos)
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_tokens, masked_pos)
        # logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)  # 两种预测

        loss_lm = criterion(logits_lm.transpose(1, 2), masked_tokens)
        loss_lm = (loss_lm.float()).mean()

        loss_clsf = criterion(logits_clsf, isNext)
        loss = loss_lm + loss_clsf

        if (epoch + 1) % 1 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        loss.backward()
        optimizer.step()

    # Inferensi contoh
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[0]
    print(text)
    print([id2vocab[w] for w in input_ids if id2vocab[w] != '[PAD]'])

    logits_lm, logits_clsf = model(torch.LongTensor([input_ids]),
                                   torch.LongTensor([segment_ids]),
                                   torch.LongTensor([masked_tokens]),
                                   torch.LongTensor([masked_pos]))
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    print('Token ter-mask (id):', [pos for pos in masked_tokens if pos != 0])
    print('Prediksi token (id):', [pos for pos in logits_lm if pos != 0])

    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
    print('Label NSP (benar):', bool(isNext))
    print('Prediksi NSP:', bool(logits_clsf))
