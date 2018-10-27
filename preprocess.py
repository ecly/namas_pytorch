"""
Preprocessing of the newsroom data.

Since the newsroom dataset is not annotated, we use nltk for tokenization.

Based on:
    https://github.com/facebookarchive/NAMAS/blob/master/dataset/process_agiga.py
"""
import sys
import re

from nltk.tokenize.treebank import TreebankWordTokenizer
import nltk.data

import jsonl

TOKENIZER = TreebankWordTokenizer()

# Words that should not be present in title according to NAMAS
# As seen in filter.py - removed those that were specific for the dataset
BAD_WORDS = {'?', 'i', ':', '-', 'by'}
def filter_instance(instance):
    title, article = instance.split("\t")
    title_words = title.split()
    article_words = article.split()
    if any((word in ("", ".", "") for word in article_words)):
        return True

    if not (10 < len(article_words) < 100 and 3 < len(title_words) < 50):
        return True

    if any(word in BAD_WORDS for word in title_words):
        return True

    # overlap between words
    matches = len({w.lower() for w in title_words if len(w) > 3} &
                  {w.lower() for w in article_words if len(w) > 3})

    if matches < 1:
        return True

    return False

# replace single digits with '#'
def remove_digits(text):
    return re.sub(r'\d', '#', text)

def parse(text):
    words = TOKENIZER.tokenize(text.lower())
    # token filter defined according to pull.py from namas
    filter_ = {'"', "'", "''", "!", "=", "-", "--", ",", "?",
               ".", "``", "`", "-rrb-", "-llb-", "\\/"}
    filtered = [w for w in words if w not in filter_]
    return " ".join(filtered)

def prepare_data_first_sentence(in_file):
    """
    This version is similar to the parsing of DUC from NAMAS.
    https://github.com/facebookarchive/NAMAS/blob/master/DUC/make_DUC.py
    <title>\t<first_sentence>\n
    """
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    with jsonl.open(in_file, gzip = True) as inp:
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

                yield parse(entry["title"]) + "\t" + parse(first)


def prepare_data(in_file):
    """
    Writes pairs in the form of title and summary as:
    <title>\t<summary>\n
    """
    with jsonl.open(in_file, gzip = True) as inp:
        for entry in inp:
            # For self-scraped sets, some things may be missing
            if entry["title"] and entry["summary"]:
                yield parse(entry["title"]) + "\t" + parse(entry["summary"])


def main():
    with open(sys.argv[2], "a") as out:
        filtered = 0
        for instance in prepare_data_first_sentence(sys.argv[1]):
            if not filter_instance(instance):
                print(instance, file=out)
            else:
                filtered += 1

        print("filtered %d instances" % filtered)


if __name__ == "__main__":
    main()

