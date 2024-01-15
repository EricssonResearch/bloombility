import os
import csv
import datetime
import random
import warnings

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from collections.abc import Iterable
from typing import Any

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
    def __init__(
        self,
        numClients,
        data_split_config=None,
        data_split="iid",
        visualize=False,
        dataset="CIFAR10",
    ):
        """
        Process:
        - Loads entire CIFAR10 trainset and testset
        - then splits the trainset into multiple trainloaders

        - Stores it if it does not already exists
            - Note: naming is train_dataset4_4 and thus does not account for different split types



        Params:
            numClients: number of clients. Indicates how many training datasets need to be created
            data_split_config: class object with hydra configuration
            data_split: what type of iid or non-iid data split shall be applied
            visualize: whether to visualize the distribution of the data split
            dataset: which dataset to use (CIFAR10 or FEMNIST). Defaults to CIFAR10

        """
        # Dictionary mapping dataset name to dataset class
        self.DATASETS = {"CIFAR10": CIFARTEN, "FEMNIST": FEMNIST}
        self.DatasetClass = self.DATASETS[dataset]

        self.num_clients = numClients

        print("Load dataset...")
        trainsets, testset = self.load_datasets()

        if data_split == "iid":
            self.trainloaders, self.testloader = self.split_dataset(
                trainsets, testset, 32
            )
        elif data_split == "num_samples" and data_split_config is not None:
            alpha = data_split_config.dirichlet_alpha
            # vvv this is the new loader that returns random number of samples for each client vvv
            self.trainloaders, self.testloader = self.split_random_size_datasets(
                trainsets, testset, 32, alpha, visualize=visualize
            )
        # vvv this is the new n-class loader that creates subsets with n classes per client vvv
        elif data_split == "num_classes" and data_split_config is not None:
            niid_factor = data_split_config.niid_factor
            self.trainloaders, self.testloader = self.split_n_classes_datasets(
                trainsets, testset, 32, niid_factor
            )
        elif data_split == "updated_non_iid" and data_split_config is not None:
            alpha = data_split_config.dirichlet_alpha
            alpha_factors = [alpha for y in range(self.num_clients)]
            self.trainloaders, self.testloader = self.split_non_iid_clients(
                trainsets,
                testset,
                self.num_clients,
                10,
                32,
                alpha_factors,
                plot=visualize,
            )
        elif (
            data_split == ("num_classes" or "num_samples" or "updated_non_iid")
            and data_split_config is None
        ):
            print("Please provide a data split config!")
            quit()

        # Store all datasets
        testset_name = "test_dataset"
        self.store_dataset(testset_name, self.testloader)
        for i in range(self.num_clients):
            trainset_name = f"{data_split}_train_dataset{i+1}_{self.num_clients}"
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
        # Define a transform to normalize the data
        if self.DatasetClass == CIFARTEN:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif self.DatasetClass == FEMNIST:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

        trainset = self.DatasetClass(
            ".", train=True, download=True, transform=transform
        )
        testset = self.DatasetClass(
            ".", train=False, download=True, transform=transform
        )

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

    def split_random_size_datasets(
        self, trainset, testset, batch_size, alpha, visualize=False
    ):
        """
        Splits the trainset into randomly sized subsets
        and then puts the trainsets and testset into DataLoaders.
        Based on num_clients, the number of clients, there are n splits of the trainset.

        Args:
            trainset: a raw trainset of maximally available length
            testset: a raw testset
            batch_size: decides the batch size for when creating the DataLoader.
            alpha: unbalancing of the dirichilet distribution
        Returns:
            trainloaders: list of DataLoader objects for training purposes
            testloader: a single DataLoader object for testing purposes
        """
        # Split training set into `num_clients` partitions to simulate different local datasets
        # generate num_clients random numbers with dirichilet distribution
        done = False
        count = 0
        while not done:
            rand_nums = np.random.dirichlet(np.ones(self.num_clients) * alpha)
            too_small = False
            for num in rand_nums:
                if num < 1 / len(trainset):
                    too_small = True
            if too_small:
                done = False
            else:
                done = True
            count = count + 1

        print(f"Generating distribution took {count} tries")

        datasets = random_split(trainset, rand_nums, torch.Generator().manual_seed(42))

        # Split into partitions and put int DataLoader as with iid
        trainloaders = []
        sizes = []

        for i, ds in enumerate(datasets):
            trainloaders.append(DataLoader(ds, batch_size=batch_size, shuffle=True))
            print(f"Size of dataloader {i+1}/{self.num_clients}: {len(ds)} images")
            sizes.append(len(ds))
        testloader = DataLoader(testset, batch_size=batch_size)

        if visualize:
            cur_date = datetime.datetime.now()
            cur_date = cur_date.strftime("%d %B %Y at %H:%M")
            sizes.insert(0, cur_date)
            sizes.insert(1, alpha)
            exp_folder = os.path.join(ROOT_DIR, "load_data", "experiments")
            if not os.path.exists(exp_folder):
                os.makedirs(exp_folder)
            file_location = os.path.join(exp_folder, "dataset_sizes.csv")
            mode = "a"
            if not os.path.exists(file_location):
                mode = "w"
            with open(file_location, mode, newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(sizes)
                csvfile.close()

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

    def split_non_iid_clients(
        self,
        trainset,
        testset,
        num_clients,
        num_classes,
        batch_size,
        alpha_factors,
        plot=False,
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
            self.make_graph(absolute_amounts, 4, 10)

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
                random_indices = self.n_rand_numbers(
                    0, len(split_arrays[i + 1][0]), overall_chunk_splits[i][k]
                )
                # print(f"Random indices of classes {i}: total len of {len(random_indices)}, array: {random_indices}")
                per_client_randoms.append(random_indices)
            per_class_randoms.append(per_client_randoms)

        # re-arrange them to be per-client instead of per-class
        transposed_list = self.transpose(per_class_randoms)

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

    def n_rand_numbers(self, start, end, num):
        """Generate num random numbers in range from start to end"""
        res = []

        for j in range(num):
            res.append(random.randint(start, end))

        return res

    def transpose(self, two_dim_list):
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

    def make_graph(self, dist, num_clients, num_classes):
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
