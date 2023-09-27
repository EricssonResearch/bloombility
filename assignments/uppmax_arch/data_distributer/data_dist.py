import sys
import os
import csv

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

'''
Program to download the CIFAR-10 dataset, split in into a number och clients and 
store all train, test and evaluation datasets seperatly in files.

MOTIVATION AND PURPOSE:
The tutorial on the Flower page downloads the data and splits it into
torch dataset objects, however, these objects needs to be distributed to
more clients. The only solution for this is either to let the server spawn
each client as a thread - or (which this solution entails) store each split
into a file and make each client know which dataset to use, which can easily
be achived by making each "client.py" take its index as a execution argument.
'''

def load_datasets(num_clients: int):
    '''
    For loading the CIFAR-10 dataset 
    #TODO: Change to actual dataset to use in production.
    '''
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    return trainloaders, valloaders, testloader


def store_dataset(dataset_name, train, evaluation):
    '''
    Stores each split of train and evaluation set to files on disk.
    '''
    n_train = len(train.dataset)
    n_eval = len(evaluation.dataset)
    print(f'Store train and eval dataset {dataset_name} (train:{n_train}, eval:{n_eval}) ...')
    # Write train dataset
    train_filename = f'train_dataset{dataset_name}.pth'
    if os.path.exists(train_filename):
        print("Train dataset already exists!")
    else:
        torch.save(train, train_filename)
    # Write eval dataset
    eval_filename = f'eval_dataset{dataset_name}.pth'
    if os.path.exists(eval_filename):
        print("Eval dataset already exists!")
    else:
        torch.save(evaluation, eval_filename)

# Check whether the correct number of arguments is supplied and then load dataset
if len(sys.argv) == 2:
        NUM_CLIENTS = int(sys.argv[1])
        print("Load dataset...")
        trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)
        # Store all splits
        for i in range(NUM_CLIENTS):
            dataset_name = f'{i}_{NUM_CLIENTS}'
            store_dataset(dataset_name, trainloaders[i], valloaders[i])
        # Store the test dataset
        print("Store test dataset...")
        if os.path.exists('test_dataset.pth'):
            print("Test dataset already exists!")
        else:
            torch.save(testloader, f'test_dataset.pth')

else:
    raise Exception("Program excepts one argument, the number of clients!")
