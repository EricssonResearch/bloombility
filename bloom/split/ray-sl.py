from copy import deepcopy
from typing import List
from torch import nn
from torch.nn import Module
from torch.optim import SGD
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.nn import CrossEntropyLoss
import numpy as np

from bloom import load_data
from bloom.models import CNNWorkerModel, CNNHeadModel

from bloom.load_data.data_distributor import DATA_DISTRIBUTOR

import argparse
import os

os.environ["RAY_DEDUP_LOGS"] = "0"  # Disable deduplication of RAY logs
import ray

# from client import WorkerModelRemote
# from server import HeadModelLocal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@ray.remote  # Specify the number of GPUs the actor should use
class ServerActor:
    def __init__(self):
        self.model = CNNHeadModel()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
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


@ray.remote  # Specify the number of GPUs the actor should use
class WorkerActor:
    def __init__(self, train_data, test_data, input_layer_size):
        self.model = CNNWorkerModel(input_layer_size)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )

    def train(self, server_actor, epochs):
        for epoch in range(epochs):
            for inputs, labels in self.train_data:
                inputs = inputs
                self.optimizer.zero_grad()
                client_output = self.model(inputs)
                grad_from_server, loss = ray.get(
                    server_actor.process_and_update.remote(client_output, labels)
                )
                client_output.backward(grad_from_server)
                self.optimizer.step()
            print(f"Epoch {epoch} completed")

    def test(self, server_actor):
        total = 0
        correct = 0
        total_loss = 0.0
        for inputs, labels in self.test_data:
            inputs = inputs
            client_output = self.model(inputs)
            loss, correct_pred, total_pred = ray.get(
                server_actor.validate.remote(client_output, labels)
            )
            total += total_pred
            correct += correct_pred
            total_loss += loss
        avg_loss = total_loss / len(self.test_data)
        accuracy = 100 * correct / total
        return avg_loss, accuracy


def main():
    num_workers = 2
    data_distributor = None
    if data_distributor is None:
        data_distributor = DATA_DISTRIBUTOR(num_workers)
        trainloaders = data_distributor.get_trainloaders()
        test_data = data_distributor.get_testloader()

    # Shut down Ray if it has already been initialized
    if ray.is_initialized():
        ray.shutdown()
    # Instantiate the server and models
    ray_ctx = ray.init(namespace="split_learning", num_cpus=num_workers + 1)

    print("============================== INFO ==============================")
    main_node_address = ray_ctx.address_info["redis_address"]
    print(f"Ray initialized with address: {main_node_address}")
    cluster_resources = ray.cluster_resources()
    print(f"Ray initialized with resources: {cluster_resources}")
    print("============================== END ==============================")

    # Create server and worker actors
    server = ServerActor.remote()

    input_layer_size = 3072
    workers = [
        WorkerActor.options(name=f"worker_{i}", namespace="split_learning").remote(
            trainloaders[i], test_data, input_layer_size
        )
        for i in range(num_workers)
    ]

    # Start training on each worker
    num_epochs = 10
    train_futures = [worker.train.remote(server, num_epochs) for worker in workers]
    ray.get(train_futures)  # Wait for training to complete

    # Start testing on each worker
    test_futures = [worker.test.remote(server) for worker in workers]
    test_results = ray.get(test_futures)

    # Aggregate test results
    avg_loss = sum([result[0] for result in test_results]) / len(test_results)
    avg_accuracy = sum([result[1] for result in test_results]) / len(test_results)
    print(f"Average Test Loss: {avg_loss}, Average Accuracy: {avg_accuracy}%")


if __name__ == "__main__":
    main()
