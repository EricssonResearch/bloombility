import os

import math
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from collections.abc import Iterable
from typing import Any

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


class DatasetSplit(Dataset):
    """Implementation of PyTorch key-value based Dataset.
    Code from https://github.com/torchfl-org/torchfl/blob/master/torchfl/datamodules/cifar.py
    """

    def __init__(self, dataset: Any, idxs: Iterable[int]) -> None:
        """Constructor

        Args:
            - dataset (Dataset): PyTorch Dataset.
            - idxs (List[int]): collection of indices.
        """
        super().__init__()
        self.dataset: Dataset = dataset
        self.idxs: list[int] = list(idxs)
        all_targets: np.ndarray = (
            np.array(dataset.targets)
            if isinstance(dataset.targets, list)
            else dataset.targets.numpy()
        )
        self.targets: np.ndarray = all_targets[self.idxs]

    def __len__(self) -> int:
        """Overriding the length method.

        Returns:
            - int: length of the collection of indices.
        """
        return len(self.idxs)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Overriding the get method.

        Args:
            - index (int): index for querying.

        Returns:
            - Tuple[Any, Any]: returns the key-value pair as a tuple.
        """
        image, label = self.dataset[self.idxs[index]]
        return image, label


class DATA_DISTRIBUTOR:
    def __init__(self, numClients, data_split="iid"):
        self.num_clients = numClients

        print("Load dataset...")
        trainsets, testset = self.load_datasets()

        if data_split == "iid":
            self.trainloaders, self.testloader = self.split_dataset(
                trainsets, testset, 32
            )
        if data_split == "num_samples":
            # vvv this is the new loader that returns random number of samples for each client vvv
            self.trainloaders, self.testloader = self.split_random_size_datasets(
                trainsets, testset, 32
            )
        # vvv this is the new n-class loader that creates subsets with n classes per client vvv
        if data_split == "num_classes":
            self.trainloaders, self.testloader = self.split_n_classes_datasets(
                trainsets, testset, 32, 2
            )

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

        Returns:
            trainset: CIFAR10 object for training purposes
            testset: CIFAR10 object for testing purposes
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
        Returns:
            trainloaders: list of DataLoader objects for training purposes
            testloader: a single DataLoader object for testing purposes
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
        Returns:
            trainloaders: list of DataLoader objects for training purposes
            testloader: a single DataLoader object for testing purposes
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

    def split_n_classes_datasets(self, trainset, testset, batch_size, niid_factor: 2):
        """
        Splits the trainset into subsets each containing niid_factor number of classes
        and then puts the trainsets and testset into DataLoaders.
        Based on num_clients, the number of clients, there are n splits of the trainset.
        Code from https://github.com/torchfl-org/torchfl/blob/master/torchfl/datamodules/cifar.py

        Comments:
        - this assumes that the number of classes has to be bigger than or equal to
            the number of shards
        - this assumes that the number of images per class in the original dataset is equal to one another
        - this assumes that the number of images per class is dividable by the shards
            - e.g. 10 images in a class, 2 classes, 5 shards
            - otherwise, the start and end indices become unaligned with the class index boundaries

        To modify:
        - split the 2D-array that has the index of the image assigned to the class along the class boundaries
        - replace the generation of ranges to take the number of images per class into account?

        Args:
            trainset: a raw trainset of maximally available length
            testset: a raw testset
            batch_size: decides the batch size for when creating the DataLoader.
            niid_factor: max number of classes held by each niid agent. Defaults to 2.
        Returns:
            trainloaders: list of DataLoader objects for training purposes
            testloader: a single DataLoader object for testing purposes
        """
        shards: int = self.num_clients * niid_factor
        # number of images per shard
        items: int = len(trainset) // shards
        # if num_clients == niid_factor ==  2, then idx_shards == [0, 1, 2, 3]
        idx_shard: list[int] = list(range(shards))
        # get every single label of the trainset as a list
        classes: np.ndarray = (
            np.array(trainset.targets)
            if isinstance(trainset.targets, list)
            else trainset.targets.numpy()
        )

        # create a 2D array that assigns the index of an image in the trainset to its class label
        idxs_labels: np.ndarray = np.vstack((np.arange(len(trainset)), classes))
        # sort based on class label in ascending order
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        # get list of indices which are sorted based on their class label
        idxs: np.ndarray = idxs_labels[0, :]
        # set up dict for datasetsplit. One dict entry per client.
        distribution: dict[int, np.ndarray] = {
            i: np.array([], dtype="int64") for i in range(self.num_clients)
        }

        while idx_shard:
            for i in range(self.num_clients):
                # parameters: 1D array to draw from, size of output shape
                #  If the given shape is, e.g., (m, n, k), then m * n * k samples are drawn
                # here: niid_factor is single int
                rand_set: set[int] = set(
                    np.random.choice(idx_shard, niid_factor, replace=False)
                )
                # remove possible shards
                idx_shard = list(set(idx_shard) - rand_set)
                for rand in rand_set:
                    # append the new shard to the original trainloader for a client:
                    # number of items in a shard, starting and ending at index given by shard
                    distribution[i] = np.concatenate(
                        (
                            distribution[i],
                            idxs[rand * items : (rand + 1) * items],
                        ),
                        axis=0,
                    )

        # make dataloaders out of it
        trainloader_list = []
        for i in distribution:
            trainloader_list.append(
                DataLoader(
                    DatasetSplit(trainset, distribution[i]),
                    batch_size=batch_size,
                    shuffle=True,
                )
            )

        testloader = DataLoader(testset, batch_size=batch_size)

        return trainloader_list, testloader

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
