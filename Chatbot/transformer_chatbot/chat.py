"""

@file  : chat.py

@author: xiaolu

@time  : 2019-12-27

"""
import pickle
import random
import time

import numpy as np
import torch

from config import Config, logger
from transformer.transformer import Transformer

if __name__ == '__main__':
    # Jalankan export.py terlebih dahulu untuk menghasilkan chatbot-v2.pt
    filename = 'chatbot-v2.pt'

    print('loading {}...'.format(filename))
    start = time.time()

    model = Transformer()
    model.load_state_dict(torch.load(filename))

    print('elapsed {} sec'.format(time.time() - start))
    model = model.to(Config.device)
    model.eval()

    logger.info('Memuat sampel uji...')
    start = time.time()
    with open(Config.data_file, 'rb') as file:
        data = pickle.load(file)
        samples = data['test']
    elapsed = time.time() - start
    logger.info('elapsed: {:.4f} seconds'.format(elapsed))

    logger.info('Memuat kosakata...')
    start = time.time()
    with open(Config.vocab_file, 'rb') as file:
        data = pickle.load(file)
        idx2char = data['dict']['idx2char']
        char2idx = data['dict']['char2idx']
    elapsed = time.time() - start
    logger.info('elapsed: {:.4f} seconds'.format(elapsed))

    print("Chatbot Transformer siap. Ketik 'q' untuk keluar.")
    while True:
        string = input("Anda: ").strip()
        if string.lower() in ('q', 'quit', 'keluar'):
            break
        sentence_in = [char2idx.get(c, Config.unk_id) for c in list(string)]

        input_sent = torch.from_numpy(np.array(sentence_in, dtype=np.long)).to(Config.device)
        input_length = torch.LongTensor([len(sentence_in)]).to(Config.device)

        with torch.no_grad():
            nbest_hyps = model.recognize(input=input_sent, input_length=input_length, char_list=idx2char)

        for hyp in nbest_hyps:
            out = hyp['yseq']
            out = [idx2char[idx] for idx in out]
            out = ''.join(out)
            out = out.replace('<sos>', '').replace('<eos>', '')

            print('Bot:', out)

