"""
Konfigurasi pelatihan klasifikasi sentimen Weibo + PEFT.
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser(description='Klasifikasi sentimen Weibo (PEFT)')
    parser.add_argument('--train_data_path', default='./data/train.csv', type=str, help='Path data latih')
    parser.add_argument('--val_data_path', default='./data/val.csv', type=str, help='Path data validasi')
    parser.add_argument('--test_data_path', default='./data/test.csv', type=str, help='Path data uji')
    parser.add_argument('--num_epochs', default=5, type=int, help='Jumlah epoch')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning rate')

    parser.add_argument('--pretrained_model_path', default='./mengzi_pretrain', type=str, help='Path model pra-latih')
    parser.add_argument('--output_dir', default='./output', type=str, help='Folder keluaran model')

    parser.add_argument('--train_batch_size', default=4, type=int, help='Ukuran batch latih')
    parser.add_argument('--val_batch_size', default=4, type=int, help='Ukuran batch validasi')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='Akumulasi gradien sebelum update')
    parser.add_argument('--seed', default=43, type=int, help='Seed acak')
    parser.add_argument('--checkpoint', default='./output/base_model.bin', type=str, help='Checkpoint terbaik')
    return parser.parse_args()
