"""
Implementasi edukatif BERT: MLM + Next Sentence Prediction (PyTorch).
Mata kuliah NLP Berbasis Transformer, Itera.
"""
import math
import re
from random import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_attn_pad_mask(seq_q, seq_k):
    """Mask perhatian: posisi PAD (id=0) tidak boleh diperhatikan."""
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    """Embedding token + posisi + segmen (tipe kalimat)."""
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(maxlen, d_model)   # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        '''
        :param x: [batch_size, max_len]
        :param seg: [batch_size, max_len]  # 0, 0, 0, 0, 1, 1, 1
        :return:
        '''
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


class ScaledDotProductAttention(nn.Module):
    """Perhatian scaled dot-product."""
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s:[batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s:[batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s:[batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # attn_mask:[batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)

        # context: [batch_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        output = nn.Linear(n_heads * d_v, d_model)(context)
        return nn.LayerNorm(d_model)(output + residual), attn   # output: [batch_size x len_q x d_model]


class PoswiseFeedForwardNet(nn.Module):
    """Position-wise feed-forward network."""
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    """Satu lapisan encoder Transformer: attention + FFN."""
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


class BERT(nn.Module):
    """BERT mini: MLM + klasifikasi pasangan kalimat (NSP)."""
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

        # Kepala NSP (next sentence)
        self.fc = nn.Linear(d_model, d_model)
        self.activ1 = nn.Tanh()
        self.classifier = nn.Linear(d_model, 2)

        # Kepala MLM (prediksi token ter-mask)
        self.linear = nn.Linear(d_model, d_model)
        self.activ2 = gelu
        self.norm = nn.LayerNorm(d_model)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)

        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output: [batch_size, len, d_model], attn: [batch_size, n_heads, d_mode, d_model]

        # NSP dari token [CLS]
        h_pooled = self.activ1(self.fc(output[:, 0]))
        logits_clsf = self.classifier(h_pooled)

        # MLM pada posisi ter-mask
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1))  # [batch_size, maxlen, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, len, d_model]
        h_masked = self.norm(self.activ2(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias  # [batch_size, maxlen, n_vocab]

        return logits_lm, logits_clsf


def make_batch():
    """Membuat satu batch contoh format BERT (MLM + NSP)."""
    batch = []
    positive = negative = 0
    while positive != batch_size/2 or negative != batch_size/2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(len(sentences))  # sample random index in sentences
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        input_ids = [vocab2id['[CLS]']] + tokens_a + [vocab2id['[SEP]']] + tokens_b + [vocab2id['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
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

        # Zero Padding (100% - 15%) tokens
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
    maxlen = 30
    batch_size = 6
    max_pred = 5
    n_layers = 6
    n_heads = 12
    d_model = 768
    d_ff = 768 * 4
    d_k = d_v = 64
    n_segments = 2

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

    token_list = list()
    for sentence in sentences:
        arr = [vocab2id[s] for s in sentence.split()]
        token_list.append(arr)

    model = BERT()
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
        torch.LongTensor(input_ids),  torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos), torch.LongTensor(isNext)

    for epoch in range(100):
        optimizer.zero_grad()
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)

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
                                   torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
    logits_lm = logits_lm.data.max(2)[1][0].data.numpy()
    print('Token ter-mask (id):', [pos for pos in masked_tokens if pos != 0])
    print('Prediksi token (id):', [pos for pos in logits_lm if pos != 0])

    logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
    print('Label NSP (benar):', bool(isNext))
    print('Prediksi NSP:', bool(logits_clsf))


