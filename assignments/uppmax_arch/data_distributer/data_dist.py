import sys
import os
import csv

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
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

def load_datasets(num_clients: int, batch_size: int):
    '''
    For loading the CIFAR-10 dataset 
    #TODO: Change to actual dataset to use in production.
    '''
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split into partitions and put int DataLoader
    trainloaders = []
    for ds in datasets:
        trainloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, testloader


def store_dataset(dataset_name, train):
    '''
    Stores each split of train dataset to files on disk.
    '''
    n_train = len(train.dataset)
    print(f'Store train dataset {dataset_name} (size:{n_train}) ...')
    # Write train dataset
    train_filename = f'train_dataset{dataset_name}.pt'
    if os.path.exists(train_filename):
        print("Train dataset already exists!")
    else:
        torch.save(train, train_filename)

# Check whether the correct number of arguments is supplied and then load dataset
if len(sys.argv) == 2:
        NUM_CLIENTS = int(sys.argv[1])
        print("Load dataset...")
        trainloaders, testloader = load_datasets(NUM_CLIENTS, 32)
        # Store all splits
        for i in range(NUM_CLIENTS):
            dataset_name = f'{i+1}_{NUM_CLIENTS}'
            store_dataset(dataset_name, trainloaders[i])
        # Store the test dataset
        print("Store test dataset...")
        if os.path.exists('test_dataset.pt'):
            print("Test dataset already exists!")
        else:
            torch.save(testloader, f'test_dataset.pt')

else:
    raise Exception("Program excepts one argument, the number of clients!")
