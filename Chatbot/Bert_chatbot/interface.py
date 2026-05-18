"""

@file  : interface.py

@author: xiaolu

@time  : 2020-03-25

"""
import torch
from seq2seq_bert import Seq2SeqModel
from bert_model import BertConfig
from tokenizer import load_bert_vocab


if __name__ == '__main__':
    word2idx = load_bert_vocab()
    config = BertConfig(len(word2idx))
    bert_seq2seq = Seq2SeqModel(config)

    # Muat model terlatih
    checkpoint = torch.load('./bert_dream.bin', map_location=torch.device("cpu"))
    bert_seq2seq.load_state_dict(checkpoint)
    bert_seq2seq.eval()

    print("Chatbot BERT siap. Ketik 'q' untuk keluar.")
    while True:
        print('Anda: ', end='')
        text = str(input()).strip()
        if text.lower() in ('q', 'quit', 'keluar'):
            break
        reply = bert_seq2seq.generate(text, beam_size=3, device='cpu')
        print('Bot: {}'.format(reply))


    # print(bert_seq2seq.generate("1", beam_size=3))
