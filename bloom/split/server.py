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


@ray.remote
class HeadModelLocal(CNNHeadModel):
    def __init__(self, lr=0.01):
        super().__init__()
        self.model = CNNHeadModel()
        self.lr = lr
        self.optimizer = SGD(self.model.parameters(), lr=0.01)
        self.loss_fn = CrossEntropyLoss()

    def getattr(self, attr):
        return getattr(self, attr)

    def logits(self, x):
        return self.model(x)

    def train_head_step(
        self,
        head_model: Module,
        input_tensor: torch.Tensor,
        labels: torch.Tensor,
        loss_fn,
        head_optimizer,
    ):
        # print("INITIATED TRAIN HEAD STEP")
        self.optimizer.zero_grad()
        # print("GETTING HEAD MODEL OUTPUT")
        output = head_model.logits.remote(input_tensor)
        ray.get(output)
        # print("HEAD MODEL OUTPUT RETRIEVED")
        # print("CALCULATING LOSS")
        loss = self.loss_fn(output, labels)
        # print("LOSS CALCULATED")
        loss.backward(retain_graph=True)
        # print("BACKWARD PROPAGATION DONE")
        head_optimizer.step()

        return input_tensor.grad, loss
