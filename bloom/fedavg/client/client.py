#!/usr/bin/env python3
import sys

# navigate to the root of the project and import the bloom package
sys.path.insert(0, "../..")  # the root path of the project
import bloom

from collections import OrderedDict
from bloom import models
import torch
import flwr as fl
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cpu")


def train(net, trainloader, epochs):
    """Train the network on the training set."""
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


def test(net, testloader):
    """Validate the network on the entire test set."""
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
    def __init__(self):
        self.net = models.FedAvgCNN().to(DEVICE)

    def load_dataset(self, train_path, test_path):
        batch_size = 32
        # Load the training dataset for this client

        self.trainloader = torch.load(train_path)
        self.testloader = torch.load(test_path)
        # Calculate the total number of samples
        num_trainset = len(self.trainloader) * batch_size
        num_testset = len(self.testloader) * batch_size
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
        train(self.net, self.trainloader, epochs=2)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def generate_client_fn(trainloaders, valloaders, num_classes):
    """Return a function that can be used by the VirtualClientEngine.

    to spawn a FlowerClient with client id `cid`.
    """

    def client_fn(cid: str):
        # This function will be called internally by the VirtualClientEngine
        # Each time the cid-th client is told to participate in the FL
        # simulation (whether it is for doing fit() or evaluate())

        # Returns a normal FLowerClient that will use the cid-th train/val
        # dataloaders as it's local data.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            num_classes=num_classes,
        )

    # return the function to spawn client
    return client_fn


"""
if __name__ == "__main__":
    # Initialize and start a single client
    if len(sys.argv) == 3:
        train_dataset_path = sys.argv[1]
        test_dataset_path = sys.argv[2]
        client = FlowerClient()
        client.load_dataset(train_dataset_path, test_dataset_path)
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    else:
        raise Exception(
            "The program expects two arguments: <train dataset file> <test dataset file>"
        )
"""
