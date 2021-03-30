import pickle
import random

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from collections import Counter

stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

def lower(s):
    """
    :param s: a string.
    return a string with lower characters
    Note that we allow the input to be nested string of a list.
    e.g.
    Input: 'Text mining is to identify useful information.'
    Output: 'text mining is to identify useful information.'
    """
    if isinstance(s, list):
        return [lower(t) for t in s]
    if isinstance(s, str):
        return s.lower()
    else:
        raise NotImplementedError("unknown datatype")


def tokenize(text):
    """
    :param text: a doc with multiple sentences, type: str
    return a word list, type: list
    e.g.
    Input: 'Text mining is to identify useful information.'
    Output: ['Text', 'mining', 'is', 'to', 'identify', 'useful', 'information', '.']
    """
    return nltk.word_tokenize(text)


def stem(tokens):
    """
    :param tokens: a list of tokens, type: list
    return a list of stemmed words, type: list
    e.g.
    Input: ['Text', 'mining', 'is', 'to', 'identify', 'useful', 'information', '.']
    Output: ['text', 'mine', 'is', 'to', 'identifi', 'use', 'inform', '.']
    """
    ### equivalent code
    # results = list()
    # for token in tokens:
    #     results.append(ps.stem(token))
    # return results

    return [ps.stem(token) for token in tokens]


def filter_stopwords(tokens):
    """
    :param tokens: a list of tokens, type: list
    return a list of filtered tokens, type: list
    e.g.
    Input: ['text', 'mine', 'is', 'to', 'identifi', 'use', 'inform', '.']
    Output: ['text', 'mine', 'identifi', 'use', 'inform', '.']
    """
    ### equivalent code
    # results = list()
    # for token in tokens:
    #     if token not in stopwords and not token.isnumeric():
    #         results.append(token)
    # return results

    return [token for token in tokens if token not in stopwords and not token.isnumeric()]

def load_data(split_name='train', columns=['text', 'stars']):
    try:
        print(f"select [{', '.join(columns)}] columns from the {split_name} split")
        df = pd.read_csv(f'data_2021_spring/{split_name}.csv')
        df = df.loc[:,columns]
        print("succeed!")
        return df
    except:
        print("Failed, then try to ")
        print(f"select all columns from the {split_name} split")
        df = pd.read_csv(f'data_2021_spring/{split_name}.csv')
        return df


def subsampling(x, y):
    labels = list(range(5))
    num = Counter(y.tolist())
    for l in labels:
        num[l] = np.sum(y == l)
    m = min(num[l] for l in labels)
    print(m, num)
    select = []
    for i in range(len(y)):
        thr = m/num[y[i]]
        if random.random() < thr:
            select.append(1)
        else:
            select.append(0)
    select_index = pd.Series(select, dtype=bool)
    _x = x[select_index].reset_index(drop=True)
    _y = y[select_index].reset_index(drop=True)
    print(Counter(_y.tolist()))
    return _x.tolist(), _y.tolist()


if __name__ == "__main__":
    train_df = load_data('train')
    train_text = train_df['text'].map(lower).map(tokenize).map(filter_stopwords)
    y_train = train_df['stars']

    word2id = {'<pad>': 0}
    for tokens in train_text:
        for t in tokens:
            if not t in word2id:
                word2id[t] = len(word2id)

    x_train = train_text.map(lambda s: [word2id.get(w, 0) for w in s])


    valid_df = load_data('valid')
    valid_text = valid_df['text'].map(lower).map(tokenize).map(filter_stopwords)
    y_valid = valid_df['stars']
    x_valid = valid_text.map(lambda s: [word2id.get(w, 0) for w in s])



    with open('data.pickle', 'rb') as f:
        x_train, y_train, x_valid, y_valid, word2id = pickle.load(f)
    # print(f"vocabulary #{len(word2id)}")
    # with open('data.pickle', 'wb') as f:
    #     pickle.dump([x_train, y_train-1, x_valid, y_valid-1, word2id], f)
    subsampling(x_train, y_train)

