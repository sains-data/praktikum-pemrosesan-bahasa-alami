"""
Konfigurasi pelatihan Transformer NMT (ZH → EN).
"""
import logging
import torch


class Config:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_file = './data/data.pkl'
    vocab_file = './data/vocab.pkl'

    train_translation_zh_filename = './data/train.zh'
    train_translation_en_filename = './data/train.en'

    valid_translation_zh_filename = './data/valid.zh'
    valid_translation_en_filename = './data/valid.en'

    n_src_vocab = 15000
    n_tgt_vocab = 15000

    pad_id = 0
    sos_id = 1
    eos_id = 2
    unk_id = 3
    IGNORE_ID = -1

    maxlen_in = 50
    maxlen_out = 100

    checkpoint = None

    grad_clip = 1.0
    print_freq = 50
    d_model = 512


def get_logger():
    """Logger konsol untuk pelatihan."""
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()
