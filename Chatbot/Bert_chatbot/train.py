"""

@file  : train.py

@author: xiaolu

@time  : 2020-03-25

"""
import torch
from torch.utils.data.dataloader import DataLoader
import time
import datetime
from seq2seq_bert import Seq2SeqModel
from bert_model import BertConfig
from dataloader import DreamDataset, collate_fn
from config import Config
from tokenizer import load_bert_vocab
from rlog import _log_normal, _log_warning, _log_info, _log_error, _log_toomuch, _log_bg_blue, _log_bg_pp, _log_fg_yl, _log_fg_cy, _log_black, rainbow


def load_model(model, pretrain_model_path):
    """Memuat bobot BERT pra-latih ke dalam model."""
    checkpoint = torch.load(pretrain_model_path)
    # Saat awal pelatihan, muat hanya lapisan bert dari checkpoint pra-latih
    checkpoint = {k[5:]: v for k, v in checkpoint.items()
                  if k[:4] == "bert" and "pooler" not in k}
    model.load_state_dict(checkpoint, strict=False)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Model BERT pra-latih berhasil dimuat.")


def train():
    # Muat dataset
    dataset = DreamDataset()
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn)

    # Inisialisasi model
    word2idx = load_bert_vocab()
    bertconfig = BertConfig(vocab_size=len(word2idx))
    bert_model = Seq2SeqModel(config=bertconfig)
    # Muat bobot pra-latih
    load_model(bert_model, Config.pretrain_model_path)
    bert_model.to(Config.device)

    # Optimizer
    optim_parameters = list(bert_model.parameters())
    optimizer = torch.optim.Adam(optim_parameters, lr=Config.learning_rate, weight_decay=1e-3)

    step = 0
    for epoch in range(Config.EPOCH):
        total_loss = 0
        i = 0
        for token_ids, token_type_ids, target_ids in dataloader:
            start_time = time.time()
            step += 1
            i += 1
            token_ids = token_ids.to(Config.device)
            token_type_ids = token_type_ids.to(Config.device)
            target_ids = target_ids.to(Config.device)
            # Dengan label target, model mengembalikan loss
            predictions, loss = bert_model(token_ids, token_type_ids, labels=target_ids, device=Config.device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time_str = datetime.datetime.now().isoformat()

            log_str = 'time:{}, epoch:{}, step:{}, loss:{:8f}, spend_time:{:6f}'.format(time_str, epoch, step, loss, time.time() - start_time)
            rainbow(log_str)
            # print('epoch:{}, step:{}, loss:{:6f}, spend_time:{}'.format(epoch, step, loss, time.time() - start_time))

            total_loss += loss.item()

            if step % 30 == 0:
                torch.save(bert_model.state_dict(), './bert_dream.bin')

        print("Epoch saat ini: {}, loss rata-rata: {:.6f}".format(epoch, total_loss / i))

        if epoch % 10 == 0:
            save_path = "./data/" + "pytorch_bert_gen_epoch{}.bin".format(str(epoch))
            torch.save(bert_model.state_dict(), save_path)
            print("{} saved!".format(save_path))


if __name__ == '__main__':
    train()


