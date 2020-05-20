import sys, os
import nltk
import scipy.sparse as sp
import numpy as np

from sklearn.preprocessing import normalize

import rouge
import re


def fetch_references(cluster):
    files = os.listdir("eval")
    files = [f for f in files if f.startswith(cluster)]
    refs = []
    for f in files:
        with open(os.path.join("eval", f), encoding="utf8") as ref_file:
            #refs.append("".join(ref_file.readlines()))
            ref_text = "".join(ref_file.readlines())
            refs.append(ref_text)

    return refs


def lines_of(c):
    files = os.listdir(c)
    for f in files:
        with open(os.path.join(c, f), encoding="utf8") as article_f:
            for l in article_f:
                yield l

def sentences_of(lines):
    start_reading = False
    text = []
    for s in lines:
        if s.strip() == "<TEXT>":
            start_reading = True
        elif s.strip() == "</TEXT>":
            start_reading = False
        elif start_reading:
            text.append(s.strip())
    text = " ".join(text)
    sents = nltk.sent_tokenize(text)
    for s in sents:
        yield s

def to_word_sequence(s):
    words = nltk.word_tokenize(s)
    words = [word.lower() for word in words if word.isalpha()]

    for w in words:
        yield w
