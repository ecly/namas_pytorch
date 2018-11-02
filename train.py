from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import sys
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import build
from model import EncoderRNN, AttnDecoderRNN

# bowDim from namas
MAX_LENGTH=50
SOS_TOKEN=0
EOS_TOKEN=1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def indexes_from_sentence(vocab, sentence):
    """Transforms a sentence to a word embedding using the given Vocabulary"""
    return [vocab.word2index[word] for word in sentence.split()]


def tensor_from_sentence(vocab, sentence, pad=False):
    """Transform a sentence to a tensor using the given Vocabulary"""
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(vocab.EOS)
    if pad:
        input_length = len(indexes)
        if input_length < MAX_LENGTH:
            padding = [SOS_TOKEN]*(MAX_LENGTH-input_length)
            indexes = padding + indexes
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(vocab, pair):
    """Transform an input and output sentence to tensors using the given Vocabulary"""
    input_tensor = tensor_from_sentence(vocab, pair[1], pad=True)
    target_tensor = tensor_from_sentence(vocab, pair[0])
    return (input_tensor, target_tensor)


def print_loss(pair_amount, loss, iter, iterations, start):
    print('Time: %s, Iteration: %d,  Avg loss: %.4f' % (timeSince(start, iter / iterations), iter, loss))


def train_iter(pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    pair_amount = len(pairs)

    for iter in range(n_iters):
        training_pair = pairs[iter % pair_amount]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH)
        print_loss_total += loss

        if iter == 0:
            continue

        if iter % pair_amount == 0:
            print('Epoch #%d' % int(iter/pair_amount))

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss(pair_amount, print_loss_avg, iter, n_iters, start)
            print_loss_total = 0



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_TOKEN:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def main():
    input_file = sys.argv[1]
    vocab = build.build_vocabulary(input_file)
    pairs = [tensors_from_pair(vocab, x.split("\t")) for x in open(input_file)]
    pairs = [(x,y) for x, y in pairs if x.size(0) <= MAX_LENGTH]

    hidden_size = 256
    encoder1 = EncoderRNN(vocab.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, vocab.n_words, dropout_p=0.1).to(device)
    train_iter(pairs, encoder1, attn_decoder1, 75000, print_every=100)


if __name__ == "__main__":
    main()
