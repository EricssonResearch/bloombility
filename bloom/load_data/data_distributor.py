import sys
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

# navigate to the root of the project and import the bloom package
# sys.path.insert(0, "../..")  # the root path of the project

from .download_cifar10 import CIFARTEN
from .download_femnist import FEMNIST
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

        # trainset = CIFAR10(".", train=True, download=True, transform=transform)
        # testset = CIFAR10(".", train=False, download=True, transform=transform)

        # trainset , testset = load_data.CIFARTEN.get_cifar10_datasets('.',transform=transform)
        trainset = FEMNIST(".", train=True, download=True, transform=transform)
        testset = FEMNIST(".", train=False, download=True, transform=transform)

        return trainset, testset

    def split_dataset(self, trainset, testset, batch_size):
        """
        Splits the trainset and then puts the trainset and dataset into DataLoaders.

        Args:
            trainset: a raw trainset
            testset: a raw testset
            num_clients: the number of clients will decide the n of splits of the trainset
            batch_size: decides the batch size for when creating the DataLoader.
        """
        # Split training set into `num_clients` partitions to simulate different local datasets
        partition_size = len(trainset) // self.num_clients
        # Calculate leftover data
        leftover = len(trainset) % self.num_clients
        # Create lengths list and distribute leftover data
        lengths = [
            partition_size + 1 if i < leftover else partition_size
            for i in range(self.num_clients)
        ]
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

        # Split into partitions and put int DataLoader
        trainloaders = []
        for ds in datasets:
            trainloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
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
        dataset_folder = os.path.join(ROOT_DIR, "load_data", "datasets")
        # Write dataset
        dataset_filename = os.path.join(dataset_folder, f"{dataset_name}.pth")
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        if os.path.exists(dataset_filename):
            print(f"{dataset_name} already exists!")
        else:
            torch.save(dataloader, dataset_filename)
