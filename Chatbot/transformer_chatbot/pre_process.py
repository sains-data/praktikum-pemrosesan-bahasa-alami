"""

@file   : pre_process.py

@author : xiaolu

@time   : 2019-12-26

"""
import pickle
import numpy as np
from tqdm import tqdm
from config import Config
from utils import encode_text


def build_vocab(token):
    """Menambahkan karakter ke kosakata jika belum ada."""
    if token not in char2idx:
        next_index = len(char2idx)
        char2idx[token] = next_index
        idx2char[next_index] = token


def process(file):
    """Memindai korpus dan membangun kosakata karakter."""
    print("processing {} ...".format(file))

    with open(file, 'r', encoding='utf8') as f:
        data = f.readlines()
    # print(data[0])   # 南京在哪里 | 在这里了

    lengths = []
    for line in tqdm(data):
        sentences = line.split('|')
        for sent in sentences:
            sentence = sent.strip()
            lengths.append(len(sentence))
            tokens = list(sentence)
            for token in tokens:
                build_vocab(token)

    np.save('./data/lengths.npy', np.array(lengths))


def get_data(in_file):
    """Membaca pasangan dialog dan mengonversi ke indeks token."""
    print('getting data {}...'.format(in_file))
    with open(in_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    samples = []
    for line in lines:
        sentences = line.split('|')
        in_sentence = sentences[0].strip()
        out_sentence = sentences[1].strip()

        in_data = encode_text(char2idx, in_sentence)
        out_data = [Config.sos_id] + encode_text(char2idx, out_sentence) + [Config.eos_id]

        if len(in_data) < Config.maxlen_in and len(out_data) < Config.maxlen_out and Config.unk_id not in in_data and Config.unk_id not in out_data:
            samples.append({'in': in_data, 'out': out_data})
    return samples


if __name__ == '__main__':
    # Bangun kosakata
    char2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    idx2char = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}

    process(Config.train_filename)

    # print("词典的大小:", len(char2idx))  # 词典的大小: 5884
    # print("前100个词:", list(char2idx.items())[:100])   # 前100个词: [('<pad>', 0), ('<sos>', 1), ('<eos>', 2)...]

    data = {
        'dict': {
            'char2idx': char2idx,
            'idx2char': idx2char
        }
    }
    with open(Config.vocab_file, 'wb') as file:
        pickle.dump(data, file)

    samples = get_data(Config.train_filename)

    np.random.shuffle(samples)
    num_samples = len(samples)
    # Validasi 1000 sampel, uji 10 sampel, sisanya latih
    num_valid = 1000
    num_test = 10

    valid = samples[:num_valid]
    test = samples[num_valid:num_valid + num_test]
    train = samples[num_valid + num_test:]

    data = {
        'train': train,
        'valid': valid,
        'test': test
    }
    print("Jumlah sampel latih:", len(train))
    print("Jumlah sampel validasi:", len(valid))
    print("Jumlah sampel uji:", len(test))

    with open(Config.data_file, 'wb') as file:
        pickle.dump(data, file)

