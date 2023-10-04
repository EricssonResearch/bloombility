#!/usr/bin/env python3
import sys

from collections import OrderedDict
import torch
import flwr as fl
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")

def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        self.net = Net().to(DEVICE)

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
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}

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
