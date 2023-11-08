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
from bloom.models import CNNHeadModel

from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
import ray
import argparse
import os
import numpy as np


class HeadModelLocal(CNNHeadModel):
    def __init__(self, input_layer_size=32, num_labels=10, lr=0.01):
        super().__init__(input_layer_size, num_labels)
        self.model = CNNHeadModel(input_layer_size, num_labels)
        self.lr = lr
        self.optimizer = SGD(self.model.parameters(), lr=0.01)

    def train_head_step(
        self,
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
