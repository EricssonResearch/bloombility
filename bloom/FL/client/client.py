#!/usr/bin/env python

from collections import OrderedDict
import torch
import flwr as fl
import wandb
from sklearn.metrics import f1_score

from bloom import models

# if you want to have metrics reported to wandb
# for each client in the federated learning
CLIENT_REPORTING = False
wandb_key = "<key>"
from bloom import ROOT_DIR
import sys
from bloom.load_data.data_distributor import DATA_DISTRIBUTOR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    batch_size = int(sys.argv[1])
    num_epochs = int(sys.argv[2])

    train_dataset_path = sys.argv[3]
    test_dataset_path = sys.argv[4]

    client = FlowerClient(batch_size, num_epochs)
    client.load_dataset(train_dataset_path, test_dataset_path)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)
    sys.stdout.flush()


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
        project="f1_non_iid_unbalancing",
    )


def train(
    net: torch.nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
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
    for i in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, train_f1 = test(
            net, trainloader
        )  # get loss and acc on train set

        if CLIENT_REPORTING:
            wandb.log(
                {
                    "dataset_len": len(trainloader),
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "epoch": i,
                }
            )

    # it can be accessed through the fit function
    # needs an aggregate_fn definition for the strategies
    # to be useful
    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_fn": train_f1,
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
    list_of_f1s = []
    net.eval()  # set to test mode
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            f1_sc = f1_score(labels, predicted, average="weighted")
            list_of_f1s.append(f1_sc)
    accuracy = correct / total
    averaged_f1 = sum(list_of_f1s) / len(list_of_f1s)

    if CLIENT_REPORTING:
        wandb.log(
            {"test_loss": loss, "test_accuracy": accuracy, "averaged_f1": averaged_f1}
        )

    print(f"average f1: {averaged_f1}")

    return loss, accuracy, averaged_f1


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, batch_size, num_epochs):
        # super().__init__()
        self.net = models.FedAvgCNN().to(DEVICE)

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        if CLIENT_REPORTING:
            wandb_login()
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

        results = train(self.net, self.trainloader, epochs=self.num_epochs)
        return self.get_parameters(config={}), self.num_examples["trainset"], results

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, f1 = test(self.net, self.testloader)
        return (
            float(loss),
            self.num_examples["testset"],
            {"accuracy": float(accuracy), "f1": f1},
        )

    def start_client(self):
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=self)


if __name__ == "__main__":
    main()
