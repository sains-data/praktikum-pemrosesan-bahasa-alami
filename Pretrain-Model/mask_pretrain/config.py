"""
Konfigurasi pra-pelatihan MLM-only (mask).
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser(description='Pra-pelatihan MLM (mask saja)')
    parser.add_argument('--train_data_path', default='./data/text.txt', type=str, help='Path data latih')
    parser.add_argument('--pretrain_weight', default='./roberta_pretrain', type=str, help='Path bobot pra-latih')
    parser.add_argument('--output_dir', default='./outputs', type=str, help='Folder keluaran model')

    parser.add_argument('--num_train_epochs', default=20, type=int, help='Jumlah epoch')
    parser.add_argument('--weight_decay_rate', default=0.01, type=float, help='Proporsi warmup scheduler')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='Epsilon AdamW')
    parser.add_argument('--train_batch_size', default=32, type=int, help='Ukuran batch latih')
    parser.add_argument('--val_batch_size', default=32, type=int, help='Ukuran batch validasi')
    parser.add_argument('--gradient_accumulation_steps', default=8, type=int, help='Akumulasi gradien')
    parser.add_argument('--learning_rate', default=5e-6, type=float, help='Learning rate')
    parser.add_argument('--seed', default=43, type=int, help='Seed acak')
    parser.add_argument('--save_checkponint_steps', default=10000, type=int, help='Simpan checkpoint tiap N step')
    return parser.parse_args()
