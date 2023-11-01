import os

import math
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# navigate to the root of the project and import the bloom package
# sys.path.insert(0, "../..")  # the root path of the project

from .download_cifar10 import CIFARTEN
from bloom import ROOT_DIR

"""
Program to download the CIFAR-10 dataset, split in into a number och clients and
store all train, test and evaluation datasets seperatly in files.

MOTIVATION AND PURPOSE:
The tutorial on the Flower page downloads the data and splits it into
torch dataset objects, however, these objects needs to be distributed to
more clients. The only solution for this is either to let the server spawn
each client as a thread - or (which this solution entails) store each split
into a file and make each client know which dataset to use, which can easily
be achived by making each "client.py" take its index as a execution argument
"""


class DATA_DISTRIBUTOR:
    def __init__(self, numClients):
        self.num_clients = numClients

        print("Load dataset...")
        trainsets, testset = self.load_datasets()
        self.trainloaders, self.testloader = self.split_dataset(trainsets, testset, 32)
        # vvv this is the new, random number of samples for each client vvv
        # self.trainloaders, self.testloader = self.split_random_size_datasets(trainsets, testset, 32)

        # Store all datasets
        testset_name = "test_dataset"
        self.store_dataset(testset_name, self.testloader)
        for i in range(self.num_clients):
            trainset_name = f"train_dataset{i+1}_{self.num_clients}"
            self.store_dataset(trainset_name, self.trainloaders[i])

    def get_trainloaders(self):
        return self.trainloaders

    def get_testloader(self):
        return self.testloader

    def load_datasets(self):
        """
        For loading a dataset (right now the CIFAR-10 dataset).
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        trainset = CIFARTEN(".", train=True, download=True, transform=transform)
        testset = CIFARTEN(".", train=False, download=True, transform=transform)

        return trainset, testset

    def split_dataset(self, trainset, testset, batch_size):
        """
        Splits the trainset into equally sized subsets
        and then puts the trainsets and testset into DataLoaders.
        Based on num_clients, the number of clients, there are n splits of the trainset.

        Args:
            trainset: a raw trainset of maximally available length
            testset: a raw testset
            batch_size: decides the batch size for when creating the DataLoader.
        """
        # Split training set into `num_clients` partitions to simulate different local datasets
        # this only works if the result is integers, not floats!
        partition_size = len(trainset) // self.num_clients
        lengths = [partition_size] * self.num_clients
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

        # Split into partitions and put int DataLoader
        trainloaders = []
        for ds in datasets:
            trainloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
        testloader = DataLoader(testset, batch_size=batch_size)
        return trainloaders, testloader

    def split_random_size_datasets(self, trainset, testset, batch_size):
        """
        Splits the trainset into randomly sized subsets
        and then puts the trainsets and testset into DataLoaders.
        Based on num_clients, the number of clients, there are n splits of the trainset.

        Args:
            trainset: a raw trainset of maximally available length
            testset: a raw testset
            batch_size: decides the batch size for when creating the DataLoader.
        """
        # Split training set into `num_clients` partitions to simulate different local datasets
        # generate num_clients random numbers with dirichilet distribution
        rand_nums = np.random.dirichlet(np.ones(self.num_clients))
        datasets = random_split(trainset, rand_nums, torch.Generator().manual_seed(42))

        # Split into partitions and put int DataLoader as with iid
        trainloaders = []
        for i, ds in enumerate(datasets):
            trainloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
            print(f"Size of dataloader {i+1}/{self.num_clients}: {len(ds)} images")
        testloader = DataLoader(testset, batch_size=batch_size)
        return trainloaders, testloader

    def store_dataset(self, dataset_name, dataloader):
        """
        Stores a dataset to disk.
        Args:
            dataset_name: a string containing the full name of the dataset
            dataloader: expects a DataLoader containing a pre-processed dataset
        """
        n_train = len(dataloader.dataset)
        print(f"Store dataset: {dataset_name} (size:{n_train}) ...")
        # Write dataset
        dataset_filename = os.path.join(
            ROOT_DIR, "load_data", "datasets", f"{dataset_name}.pth"
        )
        if os.path.exists(dataset_filename):
            print(f"{dataset_name} already exists!")
        else:
            torch.save(dataloader, dataset_filename)
