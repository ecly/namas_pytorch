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

# replace single digits with '#'
def remove_digits(text):
    return re.sub(r'\d', '#', text)

def parse(text):
    return word_tokenize(text)

def prepare_data(in_file, out_file):
    with jsonl.open(in_file, gzip = True) as inp, open(out_file, "a") as out:
        for entry in inp:
            # For self-scraped sets, some things may be missing
            if entry["title"] and entry["summary"]:
                title = parse(entry["title"])
                summary = parse(entry["summary"])
                out.write("\t".join(title) + "\n")
                out.write("\t".join(summary) + "\n\n")


if __name__ == "__main__":
    prepare_data(sys.argv[1], sys.argv[2])
