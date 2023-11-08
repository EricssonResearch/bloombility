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

# Set MAX number of clients
MAX_CLIENTS = 10


@ray.remote
def train_head_step(
    head_model: Module,
    input_tensor: torch.Tensor,
    labels: torch.Tensor,
    loss_fn,
    head_optimizer,
):
    head_optimizer.zero_grad()
    output = head_model(input_tensor)
    loss = loss_fn(output, labels)
    loss.backward()
    head_optimizer.step()

    return input_tensor.grad, loss


@ray.remote
def split_train_step(
    worker_model: Module,
    head_model: Module,
    loss_fn,
    worker_optimizer,
    head_optimizer,
    train_data: DataLoader,
    test_data: DataLoader,
):
    total_loss = 0

    worker_model.train()
    head_model.train()

    for features, labels in train_data:
        # forward propagation worker model
        worker_optimizer.zero_grad()

        cut_layer_tensor = worker_model.remote(features)

        client_output = cut_layer_tensor.clone().detach().requires_grad_(True)

        # perform forward propagation on the head model and then we receive
        # the gradients to do backward propagation
        grads, loss = train_head_step.remote(
            head_model=head_model,
            input_tensor=client_output,
            labels=labels,
            loss_fn=loss_fn,
            head_optimizer=head_optimizer,
        )

        # backward propagation on the worker node
        cut_layer_tensor.backward.remote(grads)
        worker_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_data)


# Use argparse to get the arguments from the command line
parser = argparse.ArgumentParser()
parser.add_argument("--num-clients", type=int, default=2, help="number of clients")

args = parser.parse_args()

num_clients = args.num_clients

if num_clients > MAX_CLIENTS:
    raise ValueError("Number of clients must be less than or equal to ", MAX_CLIENTS)

data_distributor = None
if data_distributor is None:
    data_distributor = DATA_DISTRIBUTOR(num_clients)
    trainloaders = data_distributor.get_trainloaders()
    testloader = data_distributor.get_testloader()

# Instantiate the server and models
ray_info = ray.init(num_cpus=num_clients)

# use python to start the server from commandline
os.system("start --head --resources='{\"server\": 1}'")
# ray start --address=<address of head node> --resources='{"worker": 100}'
os.system(
    f"start --address={ray_info['redis_address']} --resources='{'worker': {num_clients}}'"
)

head_model = CNNHeadModel()
input_layer_size = 784
worker_models = [WorkerModelRemote.remote(input_layer_size) for _ in range(num_clients)]
