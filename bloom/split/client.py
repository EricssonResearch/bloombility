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
from bloom.models import CNNWorkerModel

from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
import ray
import numpy as np


@ray.remote
class WorkerModelRemote(CNNWorkerModel):
    def __init__(self, input_layer_size):
        super().__init__(input_layer_size)
        self.model = CNNWorkerModel(input_layer_size)
        self.optimizer = SGD(self.model.parameters(), lr=0.01)
