"""
@file   : config.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-04-06
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser(description='NER GlobalPointer')
    parser.add_argument('--train_data_path', default='./data/train.json', type=str, help='Path data latih')
    parser.add_argument('--valid_data_path', default='./data/dev.json', type=str, help='Path data validasi')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='Learning rate awal')
    parser.add_argument('--output_dir', default='output', type=str, help='Folder simpan model')
    parser.add_argument('--num_epochs', default=50, type=int, help='Jumlah epoch')
    parser.add_argument('--batch_size', default=64, type=int, help='Ukuran batch')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='Langkah akumulasi gradien')
    parser.add_argument('--max_seq_len', default=64, type=int, help='Panjang sekuens maksimum')
    return parser.parse_args()
