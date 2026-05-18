"""
Konfigurasi pelatihan ClipCap (CLIP + GPT-2).
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser(description='ClipCap — CLIP prefix + GPT-2')
    parser.add_argument('--output_dir', default='output', type=str, help='Folder checkpoint dan log')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpt2_path', default='./gpt2_pretrain')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=5000)
    parser.add_argument('--dev_size', type=int, default=1000, help='Jumlah sampel validasi')
    parser.add_argument('--prefix_len', type=int, default=10, help='Panjang prefix CLIP')
    parser.add_argument('--clip_size', type=int, default=512)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--eval_step', type=int, default=10000, help='Simpan & evaluasi setiap N langkah')
    args = parser.parse_args()
    return args
