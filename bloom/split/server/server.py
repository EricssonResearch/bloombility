from typing import List
from torch import nn
from torch.nn import Module
import torch.optim as optim
from torchvision import transforms
import torch
from torch.nn import CrossEntropyLoss
from bloom.load_data.data_distributor import DATA_DISTRIBUTOR

# Import the model you want to use based on models/Networks.py
from bloom.models import CNNHeadModel
import ray
import numpy as np


# Define a dictionary mapping optimizer names to their classes
OPTIMIZERS = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}


@ray.remote  # Specify the number of GPUs the actor should use
class ServerActor:
    def __init__(self, config={}):
        self.model = CNNHeadModel()
        self.criterion = nn.CrossEntropyLoss()
        # Create the optimizer using the configuration parameters
        OptimizerClass = OPTIMIZERS[config.optimizer]
        self.optimizer = OptimizerClass(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    def process_and_update(self, client_output, labels):
        self.optimizer.zero_grad()
        labels = labels
        # print("CLIENT OUTPUT: ", client_output.shape)
        output = self.model(client_output)
        # print(output)
        # print(output.shape)
        loss = self.criterion(output, labels)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return client_output.grad, loss.item()

    def validate(self, client_output, labels):
        with torch.no_grad():
            labels = labels
            output = self.model(client_output)
            loss = self.criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
        return loss.item(), correct, total
