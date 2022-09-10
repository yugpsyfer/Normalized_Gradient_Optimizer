import os
from io import open

import nltk
nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

import wandb
import datetime

device = torch.device('cuda:0')
src_path = './ptbdataset/rnn_data/'
model_save_path = "./checkpoint/"


class Dictionary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        word = word.lower()
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self):
        self.dictionary = Dictionary()

    def make_data(self, path):
        with open(path, 'r') as fp:
            data = self.__tokenize__(fp.readlines())
        return data

    def __tokenize__(self, data):
        temp = []
        for line in data:
            for word in line.split(' '):
                temp.append(self.dictionary.add_word(word))
        return temp


def make_sequences(data, seqlen):
    temp = []
    targets = []

    total_seqs = len(data) // seqlen
    i = 0

    while total_seqs:
        temp.append(data[i:seqlen + i])
        targets.append(data[seqlen + i])
        i += 1
        total_seqs -= 1

    return temp, targets


class TextDataset(Dataset):
    def __init__(self, data):
        self.data, self.target = data

    def __len__(self):
        return len(self.target)

    def __getitem__(self, index):
        dat = torch.tensor([torch.tensor(i, dtype=torch.long) for i in self.data[index]])
        tar = torch.tensor(self.target[index], dtype=torch.long)
        return dat, tar


class TextRnn(nn.Module):
    def __init__(self, n_class, n_hidden, n_layers, n_tokens):
        super().__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden, num_layers=n_layers, batch_first = True)
        self.decoder = nn.Linear(n_hidden, n_tokens)
        self.embedder = nn.Embedding(n_tokens, n_hidden)

    def forward(self, X):
        input = self.embedder(X)
        outputs, (_, _) = self.rnn(input)
        outputs = outputs[:,-1,:]
        outputs = self.decoder(outputs)

        return outputs


def modified_train(epochs, model, beta, lr):
    for epoch in range(epochs):
        l = 0
        c = 0

        for batch in train_loader:
            data, labels = batch

            data = data.to(device)
            labels = labels.to(device)

            model.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()

            for p in model.parameters():

                if len(p.grad.shape) == 2:
                    x = p.grad
                    y = torch.norm(x, dim=1)
                    x = F.normalize(x, dim=1)
                    y = (torch.tile(y.T, dims=(p.grad.shape[1], 1))).reshape(p.grad.shape[0], p.grad.shape[1])
                else:
                    y = p.grad
                    x = torch.ones(device=device, size=p.grad.shape, dtype=torch.double)

                y[beta < y] = beta
                x = y * x

                p.data.add_(x, alpha=-lr)

            l += loss.detach().cpu()
            c += 1

        if (epoch + 1) % 5 == 0:
            vl, vp = eval(model, valid_loader)

            wandb.log({"Training_loss": l / c,
                       "Training_perplexity": torch.exp(l / c),
                       "Validation_loss": vl,
                       "Validation_perplexity": vp})

            torch.save(model, model_save_path + "rnn_best.pt")


@torch.no_grad()
def eval(model, val):
    l = 0
    c = 0
    for batch in val:
        data, labels = batch

        data = data.to(device)
        labels = labels.to(device)

        output = model(data)
        loss = criterion(output, labels)

        l += loss.detach().cpu()
        c += 1

    l = l / c

    return l, torch.exp(l)


if __name__ == "__main__":
    corpus = Corpus()

    train_data = corpus.make_data(src_path + "train.txt")
    valid_data = corpus.make_data(src_path + 'valid.txt')
    test_data = corpus.make_data(src_path + 'test.txt')

    seq_len = 35
    batch_size = 2048
    embedding_dim = 200
    n_hidden = 200
    n_layers = 4
    n_tokens = len(corpus.dictionary)
    lr = 1
    beta = 1
    epochs = 250
    criterion = nn.CrossEntropyLoss()

    config = {
        "Model": "RNN",
        "Epochs": epochs,
        "Batch Size": batch_size,
        "Sequence Length": seq_len,
        "Embedding size": embedding_dim,
        "Hidden Size": n_hidden,
        "Layers": n_layers,
        "Learning rate": lr,
        "Beta": beta,
        "Dictionary size": n_tokens
    }

    wandb.init(
        project="Normalized gradient optimizer",
        entity="yugansh",
        name="experiment_rnn_" + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        config=config
    )

    train_sequences = make_sequences(train_data, seq_len)
    test_sequences = make_sequences(test_data, seq_len)
    valid_sequences = make_sequences(valid_data, seq_len)

    train_set = TextDataset(train_sequences)
    test_set = TextDataset(test_sequences)
    val_set = TextDataset(valid_sequences)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = TextRnn(embedding_dim, n_hidden, n_layers, n_tokens)
    model.to(device)
