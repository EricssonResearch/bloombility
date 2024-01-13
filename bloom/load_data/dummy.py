import os
import csv
import datetime
import warnings
import random

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
    trainset, testset, num_clients, num_classes, batch_size, alpha_factors, plot=False
):
    """Splits non-IID client datasets from a given dataset."""

    print("Generating non-iid splits, this takes a second...")

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

    # return_index returns the indices of ar that result in the unique array == the first appearance
    _, locations, counts = np.unique(
        idxs_labels[1], return_index=True, return_counts=True
    )

    # split idxs_labels into separate arrays for each class
    # split_array[0] contains shape info, from there on is data until num_classes+1
    # every [x][1] contains the labels, [x][0] the locations
    split_arrays = np.split(idxs_labels, indices_or_sections=locations, axis=1)

    # dirichilet implementation
    all_distributions = []

    # get distribution per class and client
    for i in range(num_clients):
        rand_nums = np.random.dirichlet(np.ones(num_classes) * alpha_factors[i])
        # print(f"rand nums of client {i}: {rand_nums}, sum = {sum(rand_nums)}")
        all_distributions.append(rand_nums)

    # find biggest dist sum per class with which to scale all others
    biggest_sum = 0
    for j in range(num_classes):
        per_class_sum = 0
        for k in range(num_clients):
            per_class_sum += all_distributions[k][j]
        # print(f"per class sum of {j}: {per_class_sum}")
        if per_class_sum > biggest_sum:
            biggest_sum = per_class_sum

    # scale down if biggest is factor of >1, otherwise up
    biggest_abs_amt = 0
    if biggest_sum <= 1:
        biggest_abs_amt = math.floor(int(np.amin(counts)) * biggest_sum)
    else:
        biggest_abs_amt = math.floor(int(np.amin(counts)) / biggest_sum)

    # convert fractions into absolute amount of images
    absolute_amounts = []
    for i in range(num_clients):
        client_absolutes = []
        for dist in all_distributions[i]:
            amt = math.floor(dist * biggest_abs_amt)
            client_absolutes.append(amt)
        # print(f"Client {i} receives {client_absolutes} samples")
        absolute_amounts.append(client_absolutes)

    if plot:
        # plot dist
        make_graph(absolute_amounts, 4, 10)

    # now, split actual dataset: for each per-class dataset, split it into chunks based on absolute_amounts

    # first, separate trainset by classes, whose indices are recorded in split_arrays
    separated_class_images = []
    for i in range(num_classes):
        # print(f"split_arr of {i}[0]: {split_arrays[i+1][0]}")
        separated_class_images.append(
            torch.utils.data.Subset(trainset, split_arrays[i + 1][0])
        )

    overall_chunk_splits = []
    # calculate chunk sizes per class
    for i in range(num_classes):
        chunk_splits = []
        for k in range(num_clients):
            chunk_splits.append(absolute_amounts[k][i])
        # print(f"Class {i} is split into chunks of {chunk_splits}")
        overall_chunk_splits.append(chunk_splits)

    # generate the indices of the chunks
    per_class_randoms = []
    for i in range(num_classes):
        per_client_randoms = []
        for k in range(num_clients):
            random_indices = n_rand_numbers(
                0, len(split_arrays[i + 1][0]), overall_chunk_splits[i][k]
            )
            # print(f"Random indices of classes {i}: total len of {len(random_indices)}, array: {random_indices}")
            per_client_randoms.append(random_indices)
        per_class_randoms.append(per_client_randoms)

    # re-arrange them to be per-client instead of per-class
    transposed_list = transpose(per_class_randoms)

    # extract chunks from subsets
    all_client_datasets = []
    for client in range(num_clients):
        per_client_set = []
        for cla in range(num_classes):
            class_client_chunk = torch.utils.data.Subset(
                separated_class_images[cla], transposed_list[client][cla]
            )
            # extend here
            per_client_set.extend(class_client_chunk)
        # append here
        # as a last step, convert them to DataLoaders
        data_loader_per_client = DataLoader(
            per_client_set, batch_size=batch_size, shuffle=True
        )
        all_client_datasets.append(data_loader_per_client)

    return all_client_datasets, DataLoader(testset, batch_size=batch_size)


def n_rand_numbers(start, end, num):
    """Generate num random numbers in range from start to end"""
    res = []

    for j in range(num):
        res.append(random.randint(start, end))

    return res


def transpose(two_dim_list):
    """transpose a two-dimensional list"""
    transposed = []
    # iterate over list l1 to the length of an item
    for i in range(len(two_dim_list[0])):
        # print(i)
        row = []
        for item in two_dim_list:
            # appending to new list with values and index positions
            # i contains index position and item contains values
            row.append(item[i])
        transposed.append(row)
    return transposed


def make_graph(dist, num_clients, num_classes):
    """create graph that shows number of samples per class per client for a single experiment.
    X axis: classes
    Y axis: clients
    Size of dots: number of samples
    """
    # would throw warning about being unable to plot -Inf,
    # but plots nothing in that place instead, which is desired effect. So suppress warning.
    warnings.simplefilter("ignore", RuntimeWarning)

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

    warnings.resetwarnings()


def main():
    train, test = get_cifar10()
    num_clients = 4
    num_classes = 10
    batch_size = 32
    # every single client can have different alphas
    alphas = [0.5 for y in range(num_clients)]

    split_non_iid_clients(
        train, test, num_clients, num_classes, batch_size, alphas, plot=True
    )


if __name__ == "__main__":
    main()
