# Implement split learning in PyTorch using ray to simulate the network.

from copy import deepcopy
from typing import List
from torch import nn
from torch.nn import Module
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
from torch.nn import CrossEntropyLoss

from bloom import load_data
from bloom.models import CNNWorkerModel, CNNHeadModel

from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
import ray
import argparse
import os
import numpy as np
from .client import WorkerModelRemote
from .server import HeadModelLocal

# ENVIROMENT VARIABLES
# MAX number of clients
MAX_CLIENTS = 10

# Number of epochs to run for each worker
EPOCHS = 5

# Learning rate for training
LEARNING_RATE = 0.01


@ray.remote
def accuracy(model: Module, head_model: Module, test_loader: DataLoader):
    model.eval()
    head_model.eval()

    correct_test = 0
    total_test_labels = 0
    for input_data, labels in test_loader:
        split_layer_tensor = model(input_data)
        logits = head_model(split_layer_tensor)

        _, predictions = logits.max(1)

        correct_test += predictions.eq(labels).sum().item()
        total_test_labels += len(labels)

    test_acc = correct_test / total_test_labels
    return test_acc


@ray.remote
def split_nn(
    worker_models: List[Module],
    head_model: Module,
    head_loss_fn,
    training_sets: List[DataLoader],
    testing_sets: List[DataLoader],
    epochs: int,
    learning_rate: float,
):
    assert len(worker_models) == len(training_sets)

    # TODO: get a list of optimizers for each worker and assign them to the worker
    # similarly assign the head optimizer to the head model
    # optimizers = []

    history = {}

    for i in range(len(worker_models)):
        history[i] = {"train_acc": [], "test_acc": [], "train_loss": []}

    for i, worker_model in enumerate(worker_models):
        for e in range(epochs):
            train_loss = worker_model.split_train_step.remote(
                head_model=head_model,
                loss_fn=head_loss_fn,
                worker_optimizer=worker_model.optimizer,
                head_optimizer=head_model.optimizer,
                train_data=training_sets[i],
                test_data=testing_sets[i],
            )

            history[i]["train_loss"].append(train_loss)
            print(f"Worker {i} Epoch {e} - Training loss: {train_loss}")

    for i, worker_model in enumerate(worker_models):
        print(
            f"Worker {i} acc: {accuracy.remote(worker_model, head_model, testing_sets[i])}"
        )

    return history


def main():
    # Use argparse to get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, default=2, help="number of clients")

    args = parser.parse_args()

    num_clients = args.num_clients

    if num_clients > MAX_CLIENTS:
        raise ValueError(
            "Number of clients must be less than or equal to ", MAX_CLIENTS
        )

    data_distributor = None
    if data_distributor is None:
        data_distributor = DATA_DISTRIBUTOR(num_clients)
        trainloaders = data_distributor.get_trainloaders()
        testloader = data_distributor.get_testloader()

    # Shut down Ray if it has already been initialized
    if ray.is_initialized():
        ray.shutdown()
    # Instantiate the server and models
    ray_info = ray.init(
        address="auto", namespace="split_learning", num_cpus=num_clients
    )
    main_node_address = ray_info["redis_address"]
    print(f"Ray initialized with address: {main_node_address}")
    cluster_resources = ray.cluster_resources()
    print(f"Ray initialized with resources: {cluster_resources}")

    # # use python to start the server from commandline
    # os.system("start --head --resources='{\"server\": 1}'")
    # # ray start --address=<address of head node> --resources='{"worker": 100}'
    # os.system(
    #     f"start --address={ray_info['redis_address']} --resources='{'worker': {num_clients}}'"
    # )

    head_model = HeadModelLocal()

    # Create Ray actors for the worker models and set to ray namesapece
    input_layer_size = 784
    worker_models = [
        WorkerModelRemote.options(
            name=f"worker_{i}", namespace="split_learning"
        ).remote(input_layer_size)
        for i in range(num_clients)
    ]

    split_nn(
        worker_models=worker_models,
        head_model=head_model,
        head_loss_fn=head_model.loss_fn,
        training_sets=trainloaders,
        testing_sets=testloader,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )


if __name__ == "__main__":
    main()
