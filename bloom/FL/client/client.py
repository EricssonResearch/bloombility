#!/usr/bin/env python

from collections import OrderedDict
import torch
import flwr as fl
import wandb
import random

from bloom import models

# if you want to have metrics reported to wandb
# for each client in the federated learning
CLIENT_REPORTING = False
wandb_key = "<key here>"

DEVICE = torch.device("cpu")


def wandb_login() -> None:
    """logs into wandb and sets up a new project
    that can log metrics for each client in fn learning
    """
    if wandb.run is None:
        wandb.login(anonymous="never", key=wandb_key)
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        entity="cs_team_b",
        # keep separate from other runs by logging to different project
        project="client_reporting_fn",
    )


def train(
    net: torch.nn.Module, trainloader: torch.utils.data.DataLoader, epochs: int
) -> None:
    """Train the network on the training set.

    Params:
        net: federated Network to be trained
        trainloader: training dataset
        epochs: number of epochs in a federated learning round
    """
    net.train()  # set to train mode
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    train_loss, train_acc = test(net, trainloader)  # get loss and acc on train set

    if CLIENT_REPORTING:
        wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})

    # it can be accessed through the fit function
    # needs an aggregate_fn definition for the strategies
    # to be useful
    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
    }

    return results


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
    net.eval()  # set to test mode
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    if CLIENT_REPORTING:
        wandb.log({"test_loss": loss, "test_accuracy": accuracy})

    return loss, accuracy


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader, testloader, batch_size, num_epochs):
        # super().__init__()
        self.net = models.FedAvgCNN().to(DEVICE)
        self.trainloader = trainloader
        self.testloader = testloader
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        num_trainset = len(self.trainloader) * self.batch_size
        num_testset = len(self.testloader) * self.batch_size
        self.num_examples = {"testset": num_trainset, "trainset": num_testset}
        if CLIENT_REPORTING:
            wandb_login()

    def load_dataset(self, train_path, test_path):
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

        results = train(self.net, self.trainloader, epochs=self.num_epochs)
        return self.get_parameters(config={}), self.num_examples["trainset"], results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy)}


def generate_client_fn(trainloaders, testloader, batch_size, num_epochs):
    """Return a function that can be used by the VirtualClientEngine
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
            testloader=testloader,
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

    # return the function to spawn client
    return client_fn
