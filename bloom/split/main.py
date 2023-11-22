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
import wandb
import os
import numpy as np
from client import WorkerModelRemote
from server import HeadModelLocal

# ENVIROMENT VARIABLES
# MAX number of clients
MAX_CLIENTS = 10

# Number of epochs to run for each worker
EPOCHS = 5

# Learning rate for training
LEARNING_RATE = 0.01

wandb_track_client = False  # <-needs to be exported to yaml
wandb_track_global = False  # <-needs to be exported to yaml
wandb_key = "<key>"


def get_mnist(batch_size: int):
    # basic data transformation for MNIST
    # outputs a 28x28=784 tensors for every sample
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    # downloads the datasets
    train_set = MNIST(
        root="./data", train=True, download=True, transform=data_transform
    )
    test_set = MNIST(
        root="./data", train=False, download=True, transform=data_transform
    )

    # data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


@ray.remote
def accuracy(model: Module, head_model: Module, test_loader: DataLoader):
    model.eval.remote()
    head_model.eval()

    correct_test = 0
    total_test_labels = 0
    for input_data, labels in test_loader:
        split_layer_tensor = ray.get(model.forward.remote(input_data))
        logits = head_model(split_layer_tensor)

        _, predictions = logits.max(1)

        correct_test += predictions.eq(labels).sum().item()
        total_test_labels += len(labels)

    test_acc = correct_test / total_test_labels
    return test_acc


# @ray.remote
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
        # client reporting login
        if wandb_track_client:
            if wandb.run is None:
                wandb.login(anonymous="never", key=wandb_key)
                # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                entity="cs_team_b",
                # keep separate from other runs by logging to different project
                project="client_reporting_split",
            )

        for e in range(epochs):
            train_loss = worker_model.split_train_step.remote(
                head_model=head_model,
                loss_fn=head_loss_fn,
                worker_optimizer=worker_model.get_optimizer.remote(),
                head_optimizer=head_model.optimizer,
                train_data=training_sets[i],
                test_data=testing_sets,
            )
            train_loss = ray.get(train_loss)

            history[i]["train_loss"].append(train_loss)
            print(f"Worker {i} Epoch {e} - Training loss: {train_loss}")

            if wandb_track_client:
                wandb.log({"training loss": train_loss})

    data = []
    acc = 0
    for i, worker_model in enumerate(worker_models):
        acc = ray.get(accuracy.remote(worker_model, head_model, testing_sets))
        print(f"Worker {i} acc: {acc}")
        if wandb_track_global:
            data.append([f"worker_{i}", acc])

    if wandb_track_global:
        table = wandb.Table(data=data, columns=["worker", "accuracy"])
        wandb.log(
            {
                "worker_accuracy_bar_chart": wandb.plot.bar(
                    table, "worker", "accuracy", title="Accuracy by worker"
                )
            }
        )

    if wandb_track_client:
        wandb.finish()

    return history


def main():
    if wandb_track_global:
        wandb.login(anonymous="never", key=wandb_key)
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            entity="cs_team_b",
            project="split_reporting",
            # track hyperparameters and run metadata
            config={"method": "split", "epochs": EPOCHS},
        )

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
    # train_loader, test_loader = get_mnist(batch_size=32)
    # # split data in two
    # train_set = []
    # for train_features, train_labels in train_loader:
    #     train_set.append((train_features, train_labels))

    # test_set = []
    # for test_features, test_labels in test_loader:
    #     test_set.append((test_features, test_labels))

    # train_split = int(len(train_set) / 2)
    # test_split = int(len(test_set) / 2)

    # trainloaders = [train_set[0:train_split], train_set[train_split:]]
    # testloaders = [test_set[0:test_split], test_set[test_split:]]

    # Shut down Ray if it has already been initialized
    if ray.is_initialized():
        ray.shutdown()
    # Instantiate the server and models
    ray_ctx = ray.init(namespace="split_learning", num_cpus=num_clients + 1)

    print("============================== INFO ==============================")
    main_node_address = ray_ctx.address_info["redis_address"]
    print(f"Ray initialized with address: {main_node_address}")
    cluster_resources = ray.cluster_resources()
    print(f"Ray initialized with resources: {cluster_resources}")
    print("============================== END ==============================")

    # # use python to start the server from commandline
    # os.system("start --head --resources='{\"server\": 1}'")
    # # ray start --address=<address of head node> --resources='{"worker": 100}'
    # os.system(
    #     f"start --address={ray_info['redis_address']} --resources='{'worker': {num_clients}}'"
    # )

    head_model = HeadModelLocal()

    # Create Ray actors for the worker models and set to ray namesapece
    input_layer_size = 3072
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

    if wandb_track_global:
        wandb.finish()


if __name__ == "__main__":
    main()
