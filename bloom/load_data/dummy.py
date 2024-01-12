import os
import csv
import datetime

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from collections.abc import Iterable
from typing import Any

from download_cifar10 import CIFARTEN
from download_femnist import FEMNIST
from bloom import ROOT_DIR


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


def get_cifar10():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = CIFARTEN(".", train=True, download=True, transform=transform)
    testset = CIFARTEN(".", train=False, download=True, transform=transform)

    return trainset, testset


def split_non_iid_clients(
    trainset,
    testset,
    num_clients,
    num_classes,
    batch_size,
    alpha_factors,
    only_num_of_classes,
):
    if num_classes / num_clients < only_num_of_classes:
        print("Not enough classes for number of clients")

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

    print("#######")
    # return_index returns the indices of ar that result in the unique array == the first appearance
    _, locations, counts = np.unique(
        idxs_labels[1], return_index=True, return_counts=True
    )

    # split idxs_labels into separate arrays for each class
    # split_array[0] contains shape info, from there on is data until num_classes+1
    # every [x][1] contains the labels, [x][0] the locations
    split_arrays = np.split(idxs_labels, indices_or_sections=locations, axis=1)

    # from here, depends on use case #

    # dirichilet implementation
    all_distributions = []

    # get distribution per class and client
    for i in range(num_clients):
        rand_nums = np.random.dirichlet(np.ones(num_classes) * alpha_factors[i])
        print(f"rand nums of client {i}: {rand_nums}, sum = {sum(rand_nums)}")
        all_distributions.append(rand_nums)

    # find biggest dist sum per class with which to scale all others
    biggest_sum = 0
    for j in range(num_classes):
        per_class_sum = 0
        for k in range(num_clients):
            per_class_sum += all_distributions[k][j]
        print(f"per class sum of {j}: {per_class_sum}")
        if per_class_sum > biggest_sum:
            biggest_sum = per_class_sum

    print(f"min number of class elements: {np.amin(counts)}")
    biggest_abs_amt = 0
    if biggest_sum <= 1:
        biggest_abs_amt = math.floor(int(np.amin(counts)) * biggest_sum)
    else:
        biggest_abs_amt = math.floor(int(np.amin(counts)) / biggest_sum)

    print(
        f"biggest class percentage: {biggest_sum}, which results in {biggest_abs_amt} samples"
    )

    absolute_amounts = []
    for i in range(num_clients):
        client_absolutes = []
        for dist in all_distributions[i]:
            print(f"dist: {math.floor(dist* biggest_abs_amt)}")
            amt = math.floor(dist * biggest_abs_amt)
            client_absolutes.append(amt)
        print(f"for client {i}: use {client_absolutes}")
        absolute_amounts.append(client_absolutes)

    # double check results:
    for i in range(num_classes):
        check_sum = 0
        for j in range(num_clients):
            check_sum += absolute_amounts[j][i]
        print(f"Max was {counts[i]} got {check_sum}")

    make_graph(absolute_amounts, 4, 10)

    # now, split actual dataset: for each per-class dataset, split it into chunks based on absolute_amounts
    # then re-arrange them to be per-client instead of per-class

    # datasets = random_split(trainset, rand_nums, torch.Generator().manual_seed(42))

    # also need to adjust the plotting funct

    # n classes implementation

    # for each client, get a list of the indices we want to have
    count = 0
    all_indices = []
    for i in range(num_clients):
        per_client_indices = []
        for j in range(only_num_of_classes):
            per_client_indices.extend(split_arrays[count + 1][0])
            count += 1
        all_indices.append(per_client_indices)

    # make dataloaders out of it
    trainloader_list = []
    for i in range(len(all_indices)):
        trainloader_list.append(
            DataLoader(
                DatasetSplit(trainset, all_indices[i]),
                batch_size=batch_size,
                shuffle=True,
            )
        )

    testloader = DataLoader(testset, batch_size=batch_size)

    return trainloader_list, testloader


def make_graph(dist, num_clients, num_classes):
    """Note: throws warning about being unable to plot -Inf, but plots nothing in that place instead, which is fine"""
    max_dataset_len = 0
    for row in range(num_clients):
        # Get the dataset sizes for the current experiment
        dataset_sizes = dist[row]

        log_sizes = np.log(dataset_sizes)

        # the dataset sizes become size of each dot on the plot.
        # X and y positions are determined by experiment run number and number of datasets in the experiment
        height = [row + 1] * len(dataset_sizes)
        positions = [i + 1 for i, _ in enumerate(dataset_sizes)]
        if len(dataset_sizes) > max_dataset_len:
            max_dataset_len = len(dataset_sizes)

        # add the line of dots representing one experiment to the plot
        plt.scatter(positions, height, s=log_sizes * 10, c=positions)

    plt.title("Number of elements per class and client")
    # Set the x and y axis labels
    plt.xlabel("Classes")
    plt.ylabel("Clients")

    locs, labels = plt.yticks()  # Get the current locations and labels.
    plt.yticks(np.arange(1, num_clients + 1, step=1))

    locs, labels = plt.xticks()  # Get the current locations and labels.
    plt.xticks(np.arange(1, max_dataset_len + 1, step=1))  # Set label locations.
    # Show the plot
    plt.show()


def main():
    train, test = get_cifar10()
    num_clients = 4
    num_classes = 10
    batch_size = 32
    # every single client can have different alphas
    alphas = [0.5 for y in range(num_clients)]
    only_num_of_classes = 2

    split_non_iid_clients(
        train, test, num_clients, num_classes, batch_size, alphas, only_num_of_classes
    )


if __name__ == "__main__":
    main()
