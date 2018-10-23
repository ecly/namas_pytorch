from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import sys

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

def indexes_from_sentence(vocab, sentence):
    """Transforms a sentence to a word embedding using the given Vocabulary"""
    return [vocab.word2index[word] for word in sentence.split()]


def tensor_from_sentence(vocab, sentence):
    """Transform a sentence to a tensor using the given Vocabulary"""
    indexes = indexes_from_sentence(vocab, sentence)
    indexes.append(vocab.EOS)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(vocab, pair):
    """Transform an input and output sentence to tensors using the given Vocabulary"""
    input_tensor = tensor_from_sentence(vocab, pair[0])
    input_length = input_tensor.size()[0]
    if input_length < MAX_LENGTH:
        input_tensor
        padding = MAX_LENGTH-input_length
        input_tensor = F.pad(input_tensor, (padding, 0), "constant", SOS_TOKEN)

    target_tensor = tensor_from_sentence(vocab, pair[1])
    return (input_tensor, target_tensor)


def train_iter(pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


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

    hidden_size = 256
    encoder1 = EncoderRNN(vocab.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, vocab.n_words, dropout_p=0.1).to(device)
    train_iter(pairs, encoder1, attn_decoder1, 75000, print_every=5000)


if __name__ == "__main__":
    main()
