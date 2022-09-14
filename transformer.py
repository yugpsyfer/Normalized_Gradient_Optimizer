import os
from io import open
import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from NormGrad import NormGrad as NGD

import numpy as np
from transformers import AlbertTokenizer, AlbertForMaskedLM, AlbertConfig
import wandb
import datetime


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


def train(epochs, model, optimizer, train_loader, valid_loader):

    for epoch in range(epochs):
        l = 0
        c = 0
        train_loader.sample.set_epoch(epoch)

        for batch in train_loader:
            input_ids, attention_mask, labels = batch   # Can be an issue since earlier it was a list that was being passed

            # input_ids = input_ids.to(device)
            # attention_mask = attention_mask.to(device)
            # labels = labels.to(device)
            optimizer.zero_grad()
            output = model(input_ids=torch.squeeze(input_ids, dim=1),
                           attention_mask=torch.squeeze(attention_mask, dim=1),
                           labels=labels)
            loss = output.loss
            loss.backward()
            optimizer.step()
            # for p in model.parameters():
            #
            #     if len(p.grad.shape) == 2:
            #         x = p.grad
            #         y = torch.norm(x, dim=1)
            #         x = F.normalize(x, dim=1)
            #         y = (torch.tile(y.T, dims=(p.grad.shape[1], 1))).reshape(p.grad.shape[0], p.grad.shape[1])
            #     else:
            #         y = p.grad
            #         x = torch.ones(device=device, size=p.grad.shape, dtype=torch.double)
            #
            #     y[beta < y] = beta
            #     x = y * x
            #
            #     p.data.add_(x, alpha=-lr)

            l += loss.detach().cpu()
            c += 1

        if (epoch + 1) % 5 == 0:
            vl, vp = eval(model, valid_loader)
            #There can be issue of logging more than once since each process on both GPU's may try to log the loss
            wandb.log({"Training_loss": l / c,
                       "Training_perplexity": torch.exp(l / c),
                       "Validation_loss": vl,
                       "Validation_perplexity": vp})

            torch.save(model, model_save_path+"transformer_best.pt")

    dist.destroy_process_group()


@torch.no_grad()
def eval(model, val):
    l = 0
    c = 0
    for batch in val:
        input_ids, attention_mask, labels = batch

        # input_ids = input_ids.to(device)
        # attention_mask = attention_mask.to(device)
        # labels = labels.to(device)

        output = model(input_ids=torch.squeeze(input_ids, dim=1),
                       attention_mask=torch.squeeze(attention_mask, dim=1),
                       labels=labels)

        l += output.loss.detach().cpu()
        c += 1

    l = l / c

    return l, torch.exp(l)


def run(rank, world_size, opt, model, batch_size, tokenizer):

    # device = torch.device('cuda:0')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    print("====== PREPARING DATA ======")
    train_set = TextData(data_path + "train.txt", tokenizer)
    # test_set = TextData(data_path + "test.txt", tokenizer)
    valid_set = TextData(data_path + "valid.txt", tokenizer)
    train_sampler = DistributedSampler(train_set, drop_last=False,
                                       rank=rank, shuffle=False, num_replicas=world_size)

    val_sampler = DistributedSampler(valid_set,
                                     drop_last=False,
                                     rank=rank, shuffle=False, num_replicas=world_size)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              sampler=train_sampler, pin_memory=False,
                              num_workers=0, drop_last=False, shuffle=False)

    # test_loader = DataLoader(test_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size,
                              sampler=val_sampler, pin_memory=False,
                              num_workers=0, drop_last=False, shuffle=False)

    print("====== Starting Training ======")
    train(epochs, model, opt, train_loader, valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Arguments required for running the file")
    parser.add_argument('world_size', type=int, help="Number of nodes for MultiGpu training")
    world_size = 2
    data_path = "./ptbdataset/"  # data path
    model_save_path = "./checkpoint/"

    albert_model_configuration = AlbertConfig(
        vocab_size=30000,  # total unique tokens
        hidden_size=256,
        num_attention_heads=4,
        intermediate_size=1024,
    )

    epochs = 100
    optimizer = "NormGrad"
    batch_size = 45
    lr = 1e-4
    beta = 1  # hyperparam

    config = albert_model_configuration.to_dict()
    config["epochs"] = epochs
    config["optimizer"] = optimizer
    config["batch_size"] = batch_size
    config["learning_rate"] = lr
    config["beta"] = beta

    wandb.init(
        project="Normalized gradient optimizer",
        entity="yugansh",
        name="experiment_transformer_" + str(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        config=config
    )

    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    model = AlbertForMaskedLM(albert_model_configuration)
    # ngd = NGD(model.parameters(), lr, beta)   #Normalized gradient optimizer
    adam = optim.Adam(model.parameters(), lr)

    mp.spawn(
        run,
        args=(world_size, adam, model, batch_size, tokenizer),
        nprocs=world_size,
        join=True
    )

    # model = AlbertForMaskedLM(albert_model_configuration)
    # # device = torch.device('cuda:0')
    #
    # model.to(device)
    #
    # print("====== PREPARING DATA ======")
    # train_set = TextData(data_path + "train.txt", tokenizer)
    # # test_set = TextData(data_path + "test.txt", tokenizer)
    # valid_set = TextData(data_path + "valid.txt", tokenizer)
    # train_sampler = DistributedSampler(train_set, drop_last=False,
    #                                    rank=2, shuffle=False, num_replicas=2)
    #
    # val_sampler = DistributedSampler(valid_set,
    #                                  drop_last=False,
    #                                  rank=2, shuffle=False, num_replicas=2)
    #
    # train_loader = DataLoader(train_set, batch_size=batch_size,
    #                           sampler=train_sampler, pin_memory=False,
    #                           num_workers=0, drop_last=False, shuffle=False)
    #
    # # test_loader = DataLoader(test_set, batch_size=batch_size)
    # valid_loader = DataLoader(valid_set, batch_size=batch_size,
    #                           sampler=val_sampler, pin_memory=False,
    #                           num_workers=0, drop_last=False, shuffle=False)
    #
    # # ngd = NGD(model.parameters(), lr, beta)   #Normalized gradient optimizer
    # adam = optim.Adam(model.parameters(), lr)
    #
    # print("====== Starting Training ======")
    # train(epochs, model, adam)