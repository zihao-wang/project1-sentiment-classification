import pickle

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset

from preproc import filter_stopwords, load_data, stem, subsampling, tokenize

with open('data.pickle', 'rb') as f:
    x_train, y_train, x_valid, y_valid, word2id = pickle.load(f)


class MyDataset(Dataset):

    def pad(self, s, l):
        if len(s) < l:
            try:
                return s + [0] * (l-len(s))
            except:
                print(s)
                exit()
        else:
            return s[:l]

    def __init__(self, seq, y, tokenl):
        assert len(seq) == len(y)
        self.seq = [
            self.pad(sq, tokenl) for sq in seq
        ]
        self.y = y

    def __getitem__(self, idx):
        return np.asarray(self.seq[idx]), self.y[idx]

    def __len__(self):
        return len(self.seq)


batch_size = 32
train_loader = DataLoader(MyDataset(x_train, y_train, 256),
                          batch_size=batch_size,
                          shuffle=True)
valid_loader = DataLoader(MyDataset(x_valid, y_valid, 256),
                          batch_size=batch_size)

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=len(word2id)+1, embedding_dim=300)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=300,
                      out_channels=128,
                      kernel_size=3,
                      stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1),
            nn.MaxPool1d(kernel_size=3, stride=1),
            nn.ReLU())
        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(128, 5),
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = torch.transpose(x, 1, 2)
        x = self.cnn1(x)
        x1 = torch.max(x, dim=-1)[0]
        # x = self.cnn2(x)
        # x2 = torch.max(x, dim=-1)[0]
        x = self.mlp(x1)
        return x

model = cnn().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)
criterion = torch.nn.CrossEntropyLoss()

batch_size = 32
valid_loader = DataLoader(MyDataset(x_valid, y_valid, 128),
                        batch_size=batch_size)

_xtrain, _ytrain = subsampling(x_train, y_train)
train_loader = DataLoader(MyDataset(_xtrain, _ytrain, 128),
                        batch_size=batch_size,
                        shuffle=True)
for e in range(1, 51):

    print('epoch', e)
    model.train()
    total_acc = 0
    total_loss = 0
    total_count = 0
    with tqdm.tqdm(train_loader) as t:
        for x, y in t:
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            total_acc += (logits.argmax(1) == y).sum().item()
            total_count += y.size(0)
            total_loss += loss.item()
            optimizer.step()
            t.set_postfix({'loss': total_loss/total_count, 'acc': total_acc/total_count})

    model.eval()
    y_pred = []
    y_true = []
    with tqdm.tqdm(valid_loader) as t:
        for x, y in t:
            x = x.cuda()
            y = y.cuda()
            logits = model(x)
            loss = criterion(logits, y)
            total_acc += (logits.argmax(1) == y).sum().item()
            total_count += len(y)
            y_pred += logits.argmax(1).tolist()
            y_true += y.tolist()
    print(classification_report(y_true, y_pred))
    print("\n\n")
    print(confusion_matrix(y_true, y_pred))
    print('accuracy', np.mean(np.asarray(y_true) == np.asarray(y_pred)))
