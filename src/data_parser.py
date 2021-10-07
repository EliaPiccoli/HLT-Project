import re
import copy
import pandas as pd
import numpy as np

DATE = r'(?:\d+([\./-])\d+\1\d+|\d+\s(gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|ottobre|novembre|dicembre)\s\d+)'
HOUR = r' \d{1,2}[\.:,]\d{2} '
MAIL = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
NUMBER = r'(?:\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})|\d+)'
PUNCT = r'[,.;:]'
UNK = "<UNK>"
PAD = "<PAD>"

class Parser:
    def __init__(self):
        self.alphabet = None
        self.tok2idx = None

    def filter_string(self, s):
        s = " " + s + " "
        s = re.sub(DATE, "<DATE>", s)
        s = re.sub(HOUR, " <HOUR> ", s)
        s = re.sub(MAIL, "<MAIL>", s)
        s = re.sub(NUMBER, "<NUMBER>", s)
        return s.strip()

    def tokenize_string(self, s, maxlen=30):
        s = re.sub(PUNCT, "", s)
        tokens = s.split()
        # only the first maxlen tokens will be considered
        if len(tokens) > maxlen:
            tokens = tokens[:maxlen]
        if len(tokens) < maxlen:
            for _ in range(maxlen - len(tokens)):
                tokens.append(PAD)
        return tokens

    def tokenize_data(self, data):
        cp_data = copy.deepcopy(data)
        for i in range(len(data)):
            cp_data[i] = self.tokenize_string(self.filter_string(data[i]))
        return cp_data

    def create_vocab(self, data, tokenized=False):
        if not tokenized:
            data = self.tokenize_data(data)
        self.alphabet = set(token for elem in data for token in elem)
        self.tok2idx = {t: (i + 2) for i, t in enumerate(self.alphabet) if t != PAD}
        self.tok2idx[UNK] = 0
        self.tok2idx[PAD] = 1

    def encode_element(self, s, size=30, tokenized=False):
        if not tokenized:
            fs = self.filter_string(s)
            s = self.tokenize_string(fs, maxlen=size)
        return np.array([self.tok2idx[t] if t in self.alphabet else self.tok2idx[UNK] for t in s])

    def encode_data(self, data, size=30, tokenized=False):
        if not tokenized:
            data = self.tokenize_data(data)
        enc_data = []
        for elem in data:
            enc_data.append(self.encode_element(elem, tokenized=True))
        return np.array(enc_data)