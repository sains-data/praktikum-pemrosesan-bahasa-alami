"""

@file  : train.py

@author: xiaolu

@time  : 2020-04-01

"""
import torch
import random
from torch import nn
from torch import optim
import os
import datetime
from data_helper import loadPrepareData, trimRareWords, batch2TrainData
from seq2seq import EncoderRNN, LuongAttnDecoderRNN
from config import Config
from rlog import _log_normal, _log_warning, _log_info, _log_error, _log_toomuch, _log_bg_blue, _log_bg_pp, _log_fg_yl, _log_fg_cy, _log_black, rainbow
import time


# Token khusus
PAD_token = 0
SOS_token = 1
EOS_token = 2


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(Config.device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=Config.MAX_LENGTH):
    """Melatih satu batch."""
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(Config.device)
    lengths = lengths.to(Config.device)
    target_variable = target_variable.to(Config.device)
    mask = mask.to(Config.device)
    # print(lengths.size())   # torch.Size([5])

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Enkode
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Input awal dekoder: token SOS untuk setiap sampel dalam batch
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(Config.device)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    teacher_forcing_ratio = 0.3
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: input dekoder = label benar langkah sebelumnya
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            decoder_input = target_variable[t].view(1, -1)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        # Tanpa teacher forcing: input = prediksi langkah sebelumnya
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )

            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(Config.device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Pemangkasan gradien
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def main():
    data_path = './data/chatbot.txt'
    voc, pairs = loadPrepareData(data_path)

    # Buang pasangan yang mengandung kata di bawah ambang frekuensi
    MIN_COUNT = Config.MIN_COUNT
    pairs = trimRareWords(voc, pairs, MIN_COUNT)

    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(Config.batch_size)])
                        for _ in range(Config.total_step)]

    embedding = nn.Embedding(voc.num_words, Config.hidden_size)

    encoder = EncoderRNN(Config.hidden_size, embedding, Config.encoder_n_layers, Config.dropout)
    decoder = LuongAttnDecoderRNN(Config.attn_model, embedding, Config.hidden_size, voc.num_words, Config.decoder_n_layers, Config.dropout)

    # Optimizer
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=Config.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=Config.learning_rate * Config.decoder_learning_ratio)

    start_iteration = 1
    save_every = 4000   # Simpan checkpoint setiap N iterasi

    for iteration in range(start_iteration, Config.total_step + 1):
        training_batch = training_batches[iteration - 1]
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        start_time = time.time()
        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, Config.batch_size, Config.clip)

        time_str = datetime.datetime.now().isoformat()
        log_str = "time: {}, Iteration: {}; Percent complete: {:.1f}%; loss: {:.4f}, spend_time: {:6f}".format(time_str, iteration, iteration / Config.total_step * 100, loss, time.time() - start_time)
        rainbow(log_str)

        # Save checkpoint
        if iteration % save_every == 0:
            save_path = './save_model/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            torch.save({
                'iteration': iteration,
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(save_path, '{}_{}_model.tar'.format(iteration, 'checkpoint')))


if __name__ == '__main__':
    main()