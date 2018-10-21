"""
Preprocessing of the newsroom data.

Since the newsroom dataset is not annotated, we use nltk for tokenization.

Based on:
    https://github.com/facebookarchive/NAMAS/blob/master/dataset/process_agiga.py
"""
import sys
import jsonl
import re
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
import nltk.data
TOKENIZER = TreebankWordTokenizer()

# replace single digits with '#'
def remove_digits(text):
    return re.sub(r'\d', '#', text)

def parse(text):
    return " ".join(TOKENIZER.tokenize(text.lower()))

def prepare_data_first_sentence(in_file, out_file):
    """
    This version is similar to the parsing of DUC from NAMAS.
    https://github.com/facebookarchive/NAMAS/blob/master/DUC/make_DUC.py
    """
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    with jsonl.open(in_file, gzip = True) as inp, open(out_file, "a") as out:
        for entry in inp:
            # For self-scraped sets, some things may be missing
            if entry["title"] and entry["text"]:
                title = parse(entry["title"])
                text = entry["text"]
                sents = sent_detector.tokenize(text)
                if len(sents) == 0:
                    continue

                first = sents[0]
                if len(first) < 130 and len(sents) > 1:
                    first += sents[1]

                out.write(parse(entry["title"]) + "\n")
                out.write(parse(first) + "\n\n")


def prepare_data(in_file, out_file):
    """
    Writes pairs in the form of title and summary as:
    <title>\n
    <summary>\n
    \n
    """
    with jsonl.open(in_file, gzip = True) as inp, open(out_file, "a") as out:
        for entry in inp:
            # For self-scraped sets, some things may be missing
            if entry["title"] and entry["summary"]:
                out.write(parse(entry["title"]) + "\n")
                out.write(parse(entry["summary"]) + "\n\n")


if __name__ == "__main__":
    prepare_data_first_sentence(sys.argv[1], sys.argv[2])
