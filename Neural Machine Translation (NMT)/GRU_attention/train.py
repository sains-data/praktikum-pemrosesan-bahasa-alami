"""
@file   : train.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2021-02-21
"""
import collections
import os
import io
import torch
import torchtext.vocab as Vocab
import torch.utils.data as Data
from torch import nn
from model import Encoder, Decoder


PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'


def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    # Padding tiap urutan ke max_seq_len
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)


def build_data(all_tokens, all_seqs):
    # Bangun kosakata dan konversi ke indeks
    vocab = Vocab.Vocab(collections.Counter(all_tokens), specials=[PAD, BOS, EOS])
    indices = [[vocab.stoi[w] for w in seq] for seq in all_seqs]
    return vocab, torch.tensor(indices)


def read_data(max_seq_len):
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('./data/fr2en.txt') as f:
        lines = f.readlines()
        for line in lines:
            in_seq, out_seq = line.rstrip().split('\t')
            in_seq_tokens = in_seq.split(' ')
            out_seq_tokens = out_seq.split(' ')
            if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
                # Abaikan sampel yang melebihi max_seq_len setelah EOS
                continue
            process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
            process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
        in_vocab, in_data = build_data(in_tokens, in_seqs)
        out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, Data.TensorDataset(in_data, out_data)


def batch_loss(encoder, decoder, X, Y, loss):
    """Hitung loss teacher-forcing untuk satu batch."""
    batch_size = X.shape[0]
    enc_state = encoder.begin_state()
    enc_outputs, enc_state = encoder(X, enc_state)
    # print(enc_outputs.size())   # torch.Size([7, 2, 64])
    # print(enc_state.size())    # torch.Size([2, 2, 64])

    dec_state = decoder.begin_state(enc_state)
    dec_input = torch.tensor([out_vocab.stoi[BOS]] * batch_size)   # token awal decoder = BOS
    mask, num_not_pad_tokens = torch.ones(batch_size,), 0

    l = torch.tensor([0.0])
    for y in Y.permute(1, 0):   # iterasi per posisi waktu (teacher forcing)
        dec_output, dec_state = decoder(
            dec_input, dec_state, enc_outputs
        )
        # print(dec_output.size())    # torch.Size([2, 38])
        # print(dec_state.size())    # torch.Size([2, 2, 64])

        l = l + (mask * loss(dec_output, y)).sum()

        dec_input = y   # teacher forcing: input = token target sebenarnya
        num_not_pad_tokens += mask.sum().item()

        mask = mask * (y != out_vocab.stoi[EOS]).float()
    return l / num_not_pad_tokens


def train(encoder, decoder, dataset, lr, batch_size, num_epochs):
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, Y in data_iter:
            # print(X.size())   # torch.Size([2, 7])
            # print(Y.size())   # torch.Size([2, 7])
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        if (epoch + 1) % 10 == 0:
            print('Epoch: {}, loss: {}'.format(epoch+1, l_sum / len(data_iter)))


def translate(encoder, decoder, input_seq, max_seq_len):
    in_tokens = input_seq.split(' ')
    in_tokens = [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)

    # batch_size = 1
    enc_input = torch.tensor([[in_vocab.stoi[tk] for tk in in_tokens]])
    enc_state = encoder.begin_state()
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = torch.tensor([out_vocab.stoi[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []

    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(dim=1)
        pred_token = out_vocab.itos[int(pred.item())]

        if pred_token == EOS:
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens


if __name__ == '__main__':
    max_seq_len = 7
    # Muat data dan latih
    in_vocab, out_vocab, dataset = read_data(max_seq_len)
    embed_size, num_hiddens, num_layers = 64, 64, 2
    attention_size = 10
    drop_prob = 0.5
    lr = 0.01
    batch_size = 2
    num_epochs = 50
    encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers, drop_prob)
    decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers, attention_size, drop_prob)
    train(encoder, decoder, dataset, lr, batch_size, num_epochs)

    print('Mulai inferensi...')
    input_seq = 'ils regardent .'
    res = translate(encoder, decoder, input_seq, max_seq_len=7)
    print(res)

