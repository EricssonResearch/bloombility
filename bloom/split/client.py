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

    def get_optimizer(self):
        return self.optimizer

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
        head_model.train.remote()

        for features, labels in train_data:
            # forward propagation worker model
            worker_optimizer.zero_grad()

            cut_layer_tensor = self.model(features)

            client_output = cut_layer_tensor.clone().detach().requires_grad_(True)

            # print("ENTERING TRAIN HEAD MODEL")
            # perform forward propagation on the head model and then we receive
            # the gradients to do backward propagation
            train_head_output = head_model.train_head_step.remote(
                head_model=head_model,
                input_tensor=client_output,
                labels=labels,
                loss_fn=loss_fn,
                head_optimizer=head_optimizer,
            )

            # print("WAITING FOR TRAIN HEAD MODEL")
            grads, loss = ray.get(train_head_output)
            # backward propagation on the worker node
            # print("BACKWARD PROPAGATION ON WORKER NODE")
            cut_layer_tensor.backward(grads)
            worker_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_data)

    # def accuracy(model: Module, head_model: Module, test_loader: DataLoader):
    #     model.eval.remote()
    #     head_model.eval()

    #     correct_test = 0
    #     total_test_labels = 0
    #     for input_data, labels in test_loader:
    #         split_layer_tensor = ray.get(model.remote(input_data))
    #         logits = head_model(split_layer_tensor)

    #         _, predictions = logits.max(1)

    #         correct_test += predictions.eq(labels).sum().item()
    #         total_test_labels += len(labels)

    #     test_acc = correct_test / total_test_labels
    #     return test_acc
