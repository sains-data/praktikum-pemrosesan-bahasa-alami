"""

@file  : data_helper.py

@author: xiaolu

@time  : 2020-05-25

"""
import os
import random


def load_dataset(path_dataset):
    """Muat dataset format BIO (kata\\tlabel, baris kosong = akhir kalimat)."""
    dataset = []
    with open(path_dataset, 'r') as f:
        words, tags = [], []
        for line in f:
            if line != '\n':   # baris kosong = akhir kalimat
                line = line.strip('\n')
                word, tag = line.split('\t')
                if len(word) > 0 and len(tag) > 0:
                    word, tag = str(word), str(tag)
                    words.append(word)
                    tags.append(tag)
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
    return dataset


def save_dataset(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences, open(os.path.join(save_dir, 'tags.txt'), 'w') as file_tags:
        for words, tags in dataset:
            file_sentences.write('{}\n'.format(' '.join(words)))
            file_tags.write('{}\n'.format(' '.join(tags)))
    print("Data berhasil disimpan.")


def build_tags(data_dir, tags_file):
    """Kumpulkan semua label unik dari train/val/test."""
    data_types = ['train', 'val', 'test']
    tags = set()
    for data_type in data_types:
        tags_path = os.path.join(data_dir, data_type, 'tags.txt')
        with open(tags_path, 'r') as file:
            for line in file:
                tag_seq = filter(len, line.strip().split(' '))
                tags.update(list(tag_seq))
    tags = list(tags)
    with open(tags_file, 'w') as file:
        file.write('\n'.join(tags))
    return tags


if __name__ == '__main__':
    path_train_val = './data/msra/msra_train_bio'
    path_test = './data/msra/msra_test_bio'

    print('Memuat data...')
    dataset_train_val = load_dataset(path_train_val)
    dataset_test = load_dataset(path_test)
    print('Jumlah kalimat train+val:', len(dataset_train_val))
    print('Jumlah kalimat test:', len(dataset_test))
    print('Pemuatan data selesai.')

    order = list(range(len(dataset_train_val)))
    random.seed(2020)
    random.shuffle(order)

    # Split train / validation
    train_dataset = [dataset_train_val[idx] for idx in order[:42000]]
    val_dataset = [dataset_train_val[idx] for idx in order[42000:]]
    test_dataset = dataset_test

    # Simpan ke folder train/val/test
    save_dataset(train_dataset, 'data/msra/train')
    save_dataset(val_dataset, 'data/msra/val')
    save_dataset(test_dataset, 'data/msra/test')

    # Build tags from dataset
    build_tags('data/msra', 'data/msra/tags.txt')

