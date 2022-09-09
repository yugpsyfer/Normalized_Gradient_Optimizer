import os
from io import open

# import nltk
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

import numpy as np
from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertConfig

import wandb
import datetime


device = torch.device('cuda:0')
data_path = "./ptbdataset/" #data path
model_save_path = "./checkpoint/"

albert_model_configuration = AlbertConfig(
    vocab_size=30000, #total unique tokens
    hidden_size=256,
    num_attention_heads=4,
    intermediate_size=1024,
)

epochs = 100
optimizer = "NormGrad"
batch_size = 45
lr = 1
beta = 1 #hyperparam


config = albert_model_configuration.to_dict()
config["epochs"] = epochs
config["optimizer"] = optimizer
config["batch_size"] = batch_size
config["learning_rate"] = lr
config["beta"] = beta


wandb.init(
    project="Normalized gradient optimizer",
    entity="yugansh",
    name="experiment_transformer_"+str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    config=config
)

model = AlbertForMaskedLM(albert_model_configuration)
model.to(device)

tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")


class TextData(Dataset):
    def __init__(self, path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        self.max_seq_len = -1
        self.__preprocess__(path)

    def __preprocess__(self, path):
        with open(path, 'r') as fp:
            for line in fp.readlines():
                self.data.append(line)
                self.max_seq_len = max(len(line), self.max_seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ip = self.tokenizer(self.data[index], return_tensors='pt', padding='max_length',
                            max_length=min(self.max_seq_len, 512))
        ip['labels'] = ip.input_ids.detach().clone()
        ip = self.__make_rand_masks__(ip)

        return ip.input_ids, ip.attention_mask, ip.labels

    @staticmethod
    def __make_rand_masks__(ip):
        r = torch.rand(ip.input_ids.shape)
        masked = r < 0.15  # BERT SETTING
        masked = masked * (ip.input_ids != 101) * (ip.input_ids != 102)
        s = torch.flatten((masked[0]).nonzero()).tolist()
        ip.input_ids[0, s] = 103  # MASK ID

        return ip


def modified_train(epochs, model, beta, lr):

    for epoch in range(epochs):
        l = 0
        c = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            model.zero_grad()
            output = model(input_ids=torch.squeeze(input_ids, dim=1),
                           attention_mask=torch.squeeze(attention_mask, dim=1),
                           labels=labels)
            loss = output.loss
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

            torch.save(model, model_save_path+"best.pt")


@torch.no_grad()
def eval(model, val):
    l = 0
    c = 0
    for batch in val:
        input_ids, attention_mask, labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        output = model(input_ids=torch.squeeze(input_ids, dim=1),
                       attention_mask=torch.squeeze(attention_mask, dim=1),
                       labels=labels)

        l += output.loss.detach().cpu()
        c+=1

    l = l / c

    return l, torch.exp(l)


if __name__ == '__main__':

    print("====== PREPARING DATA ======")
    train_set = TextData(data_path + "train.txt", tokenizer)
    test_set = TextData(data_path + "test.txt", tokenizer)
    valid_set = TextData(data_path + "valid.txt", tokenizer)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    print("====== Starting Training ======")
    modified_train(epochs, model, lr, beta)