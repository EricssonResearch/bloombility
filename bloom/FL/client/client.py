#!/usr/bin/env python

from collections import OrderedDict
import torch
import flwr as fl

from bloom import models
import argparse
import yaml
import os
from bloom import ROOT_DIR

import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "federated")


def main():
    parser = argparse.ArgumentParser(
        prog="client.py", description="runs a flower server"
    )

    parser.add_argument("-n", "--num_clients", type=int, default=4, dest="num_clients")
    parser.add_argument(
        "-c",
        "--config",
        default="default.yaml",
        type=str,
        required=False,
        dest="config_file",
    )
    parser.add_argument("--train", required=True, type=str, dest="train_path")
    parser.add_argument("--test", required=True, type=str, dest="test_path")
    args = parser.parse_args()

    print(f'Device: {DEVICE}')

    cfg_path = os.path.join(CONFIG_PATH, args.config_file)
    with open(cfg_path, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    batch_size = cfg["client"]["hyper_params"]["batch_size"]
    num_epochs = cfg["client"]["hyper_params"]["num_epochs"]

    # Configure logging in each subprocess
    logging.basicConfig(filename="clients.log", level=logging.INFO)

    # Example log statement with explicit flushing
    logging.debug("Debug message")
    logging.getLogger().handlers[0].flush()

    train_path = args.train_path
    test_path = args.test_path

    client = FlowerClient(batch_size, num_epochs)
    client.load_dataset(train_path, test_path)
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
        self.net = models.CNNFemnist().to(DEVICE)

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
