"""
Module for building dictionaries based on the input.
"""
import sys
import jsonl


class Vocabulary:
    def __init__(self):
        self.SOS = 0
        self.EOS = 1
        self.UNK = 1
        self.word2count = {}
        self.word2index = {"<s>": self.SOS, "</s>": self.EOS, "<unk>": self.UNK}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = len(self.word2index)

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def parse_instance(instance):
    return instance.split("\t")


def build_vocabulary(input_file):
    with open(input_file, "r") as inp:
        vocab = Vocabulary()
        for inst in inp.readlines():
            title, summary = parse_instance(inst.strip())
            vocab.add_sentence(title)
            vocab.add_sentence(summary)

        return vocab


if __name__ == "__main__":
    build_vocabulary(sys.argv[1])
