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
    def __init__(self, input_layer_size, lr=0.01):
        super().__init__(input_layer_size)
        self.lr = lr
        self.model = CNNWorkerModel(input_layer_size)
        self.optimizer = SGD(self.model.parameters(), lr=self.lr)

    def split_train_step(
        self,
        head_model: Module,
        loss_fn,
        worker_optimizer,
        head_optimizer,
        train_data: DataLoader,
        test_data: DataLoader,
    ):
        total_loss = 0

        self.model.train()
        head_model.train()

        for features, labels in train_data:
            # forward propagation worker model
            worker_optimizer.zero_grad()

            cut_layer_tensor = self.model.remote(features)

            client_output = cut_layer_tensor.clone().detach().requires_grad_(True)

            # perform forward propagation on the head model and then we receive
            # the gradients to do backward propagation
            grads, loss = head_model.train_head_step(
                head_model=head_model,
                input_tensor=client_output,
                labels=labels,
                loss_fn=loss_fn,
                head_optimizer=head_optimizer,
            )

            # backward propagation on the worker node
            cut_layer_tensor.backward(grads)
            worker_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_data)
