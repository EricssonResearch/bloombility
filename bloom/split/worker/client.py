from typing import List
from torch import nn
from torch.nn import Module
import torch.optim as optim
from torchvision import transforms
import torch
from torch.nn import CrossEntropyLoss
from bloom.models import CNNWorkerModel
from bloom import ROOT_DIR
import ray
import numpy as np
import wandb

# Define a dictionary mapping optimizer names to their classes
OPTIMIZERS = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}


@ray.remote
class WorkerActor:
    def __init__(self, train_data, test_data, input_layer_size, wandb=False, config={}):
        self.model = CNNWorkerModel(input_layer_size)
        self.train_data = train_data
        self.test_data = test_data
        # Create the optimizer using the configuration parameters
        OptimizerClass = OPTIMIZERS[config.optimizer]
        self.optimizer = OptimizerClass(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.losses = []
        self.wandb = wandb

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
            if self.wandb:
                wandb.log({"loss": loss})
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

    def get_model(self):
        return self.model

    # define a function to return the model's weights
    def get_weights(self):
        return self.model.state_dict()

    # define a function to set the model's weights
    def set_weights(self, weights):
        self.model.load_state_dict(weights)
