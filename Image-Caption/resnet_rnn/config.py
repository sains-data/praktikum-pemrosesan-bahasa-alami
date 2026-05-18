"""
Konfigurasi pelatihan image captioning (ResNet + LSTM).
"""
import argparse


def set_args():
    parser = argparse.ArgumentParser(description='Image to Caption — ResNet + Attention LSTM')
    parser.add_argument('--epochs', default=200, type=int, help='Jumlah epoch pelatihan')
    parser.add_argument('--max_len', default=40, type=int, help='Panjang maksimum caption')
    parser.add_argument('--batch_size', default=64, type=int, help='Ukuran batch')

    parser.add_argument('--alpha_c', default=1., type=float, help='Bobot regularisasi attention ganda stokastik')

    parser.add_argument('--decoder_lr', default=1e-4, type=float, help='Learning rate decoder')
    parser.add_argument('--encoder_lr', default=4e-4, type=float, help='Learning rate encoder')

    parser.add_argument('--grad_clip', default=5., type=float, help='Norm gradien maksimum (clipping)')
    parser.add_argument('--print_freq', default=1000, type=int, help='Frekuensi log validasi (batch)')

    parser.add_argument('--attention_dim', default=512, type=int, help='Dimensi lapisan attention')
    parser.add_argument('--emb_dim', default=512, type=int, help='Dimensi embedding kata')
    parser.add_argument('--decoder_dim', default=512, type=int, help='Dimensi hidden LSTM decoder')
    parser.add_argument('--dropout', default=0.5, type=float, help='Dropout decoder')

    args = parser.parse_args()
    return args
