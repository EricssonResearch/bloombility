#!/usr/bin/env python

from collections import OrderedDict
import torch
import flwr as fl

from bloom import models
from bloom import ROOT_DIR
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import sys
from bloom.load_data.data_distributor import DATA_DISTRIBUTOR


import logging

config_path = os.path.join(ROOT_DIR, "config", "federated")
DEVICE = torch.device("cuda")


@hydra.main(config_path=config_path, config_name="base", version_base=None)
def main(cfg: DictConfig):
    batch_size = cfg.client.hyper_params.batch_size
    num_epochs = cfg.client.hyper_params.num_epochs

    # Configure logging in each subprocess
    logging.basicConfig(filename="clients.log", level=logging.INFO)

    # Example log statement with explicit flushing
    logging.debug("Debug message")
    logging.getLogger().handlers[0].flush()

    train_dataset_path = sys.argv[3]
    test_dataset_path = sys.argv[4]

    client = FlowerClient(batch_size, num_epochs)
    client.load_dataset(train_dataset_path, test_dataset_path)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


def train(
    net: torch.nn.Module, trainloader: torch.utils.data.DataLoader, epochs: int
) -> None:
    """Train the network on the training set.

    Params:
        net: federated Network to be trained
        trainloader: training dataset
        epochs: number of epochs in a federated learning round
    """
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters())
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(
    net: torch.nn.Module, testloader: torch.utils.data.DataLoader
) -> tuple[float, float]:
    """Validate the network on the entire test set.

    Calculates classification accuracy & loss.
    Params:
        net: Network to be tested
        testloader: test dataset to evaluate Network with
    Returns:
        loss: difference between expected and actual result
        accuracy: accuracy of network on testing dataset
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    return loss, accuracy


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, batch_size, num_epochs):
        # super().__init__()
        self.net = models.FedAvgCNN().to(DEVICE)

        self.batch_size = batch_size
        self.num_epochs = num_epochs

        print("flower client created..")

    def load_dataset(self, train_path, test_path):
        self.trainloader = torch.load(train_path)
        self.testloader = torch.load(test_path)

        # Calculate the total number of samples
        num_trainset = len(self.trainloader) * self.batch_size
        num_testset = len(self.testloader) * self.batch_size
        self.num_examples = {"testset": num_trainset, "trainset": num_testset}

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    # update the local model weights with the parameters received from the server
    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=self.num_epochs)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

    def start_client(self):
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=self)


if __name__ == "__main__":
    main()
