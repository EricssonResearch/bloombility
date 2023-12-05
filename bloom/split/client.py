from typing import List
from torch import nn
from torch.nn import Module
import torch.optim as optim
from torchvision import transforms
import torch
from torch.nn import CrossEntropyLoss
from bloom.models import CNNWorkerModel
import ray
import numpy as np


@ray.remote
class WorkerActor:
    def __init__(self, train_data, test_data, input_layer_size):
        self.model = CNNWorkerModel(input_layer_size)
        self.train_data = train_data
        self.test_data = test_data
        self.optimizer = optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        self.losses = []

    def getattr(self, attr):
        return getattr(self, attr)

    def train(self, server_actor, epochs):
        for epoch in range(epochs):
            loss = 0.0
            for inputs, labels in self.train_data:
                inputs = inputs
                self.optimizer.zero_grad()
                client_output = self.model(inputs)
                grad_from_server, loss = ray.get(
                    server_actor.process_and_update.remote(client_output, labels)
                )
                client_output.backward(grad_from_server)
                self.optimizer.step()
            self.losses.append(loss)
            print(f"Epoch {epoch+1} completed, loss: {loss}")

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
