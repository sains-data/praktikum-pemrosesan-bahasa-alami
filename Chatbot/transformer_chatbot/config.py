"""

@file   : config.py

@author : xiaolu

@time   : 2019-12-26

"""
import logging
import torch


class Config:
    # Perangkat: GPU jika tersedia
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Path korpus mentah (format: pertanyaan|jawaban per baris)
    train_filename = './data/12万对话语料青云库.csv'

    # Kosakata dan data terproses
    vocab_file = './data/vocab.pkl'
    data_file = './data/data.pkl'

    # Token khusus
    pad_id = 0
    sos_id = 1
    eos_id = 2
    unk_id = 3
    IGNORE_ID = -1

    maxlen_in = 50
    maxlen_out = 50

    # Ukuran kosakata (perbarui setelah pre_process.py)
    vocab_size = 5884

    grad_clip = 1.0  # clip gradients at an absolute value of (-1, 1)

    print_freq = 50

    checkpoint = None

    d_model = 512


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(threadName)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger()