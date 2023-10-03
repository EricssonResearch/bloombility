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

def load_datasets():
    """
    For loading a dataset (right now the CIFAR-10 dataset). 
    """
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)

    return trainset, testset

def split_dataset(trainset, testset, num_clients, batch_size):
    """
    Splits the trainset and then puts the trainset and dataset into DataLoaders.
    
    Args:
        trainset: a raw trainset
        testset: a raw testset
        num_clients: the number of clients will decide the n of splits of the trainset
        batch_size: decides the batch size for when creating the DataLoader.
    """    
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


def store_dataset(dataset_name, trainset):
    """
    Stores each split of train dataset to files on disk.
    Args:
        dataset_name: a string containing the full name of train data-subset
        trainset: expects a DataLoader containing the trainset
    """
    n_train = len(trainset.dataset)
    print(f'Store train dataset {dataset_name} (size:{n_train}) ...')
    # Write train dataset
    train_filename = f'datasets/train_dataset{dataset_name}.pth'
    if os.path.exists(train_filename):
        print("Train dataset already exists!")
    else:
        torch.save(trainset, train_filename)

# Check whether the correct number of arguments is supplied and then load dataset
if len(sys.argv) == 2:
        num_clients = int(sys.argv[1])
        print("Load dataset...")
        trainsets, testset = load_datasets()
        trainloaders, testloader = split_dataset(trainsets, testset, num_clients, 32)
        # Store all splits
        for i in range(num_clients):
            dataset_name = f'{i+1}_{num_clients}'
            store_dataset(dataset_name, trainloaders[i])
        # Store the test dataset
        print("Store test dataset...")
        test_filename = "datasets/test_dataset.pth"
        if os.path.exists(test_filename):
            print("Test dataset already exists!")
        else:
            torch.save(testloader, test_filename)

else:
    raise Exception("Program excepts one argument, the number of clients!")
