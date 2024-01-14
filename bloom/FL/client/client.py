#!/usr/bin/env python

from collections import OrderedDict
import torch
import flwr as fl

from bloom import models
import argparse
import yaml
import os
from bloom import ROOT_DIR
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import wandb
from datetime import datetime

import logging

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "federated")
CLIENT_REPORTING = False


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

    print(f"Device: {DEVICE}")

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
    print(f"Training started for {epochs} epochs")
    for i in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

        train_loss, train_acc, train_f1, precision, recall = test(
            net, trainloader
        )  # get loss and acc on train set
        print("Epoch: %d Finished" % (i + 1))
    print(f"Training completed for {epochs} epochs")

    # it can be accessed through the fit function
    # needs an aggregate_fn definition for the strategies
    # to be useful
    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_fn": train_f1,
    }

    return results


def test(net: torch.nn.Module, testloader: torch.utils.data.DataLoader):
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

    labels_list = []
    pred_list = []
    all_scores = []
    all_labels = []
    net.eval()  # set to test mode
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Move tensors to CPU before using them with NumPy
            labels_list.extend(labels.cpu().numpy())
            pred_list.extend(predicted.cpu().numpy())

            scores = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            all_scores.extend(scores)
            _labels = label_binarize(labels, classes=list(range(62)))
            all_labels.extend(_labels)

    accuracy = correct / total

    f1 = f1_score(labels_list, pred_list, average="micro")
    precision = precision_score(labels_list, pred_list, average="micro")
    recall = recall_score(labels_list, pred_list, average="micro")

    # Concatenate all the scores and labels
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    return loss, accuracy, f1, precision, recall, all_scores, all_labels


def plot_precision_recall(
    y_test,
    y_score,
    wandb_track: bool = False,
    showPlot: bool = False,
):
    precision = dict()
    recall = dict()
    average_precision = dict()
    # Number of classes (10 for CIFAR-10, 62 for FEMNIST)
    n_classes = 62

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_test, y_score, average="micro"
    )

    # Plot the micro-averaged Precision-Recall curve
    plt.figure(figsize=(6.4 * 2, 4.8 * 2))
    plt.plot(
        recall["micro"],
        precision["micro"],
        color="gold",
        lw=2,
        label="micro-average (area = {0:0.2f})" "".format(average_precision["micro"]),
    )

    # # Dedine list of colors for plotting
    # colors = cycle(
    #     [
    #         "aqua",
    #         "darkorange",
    #         "cornflowerblue",
    #         "red",
    #         "green",
    #         "blue",
    #         "yellow",
    #         "purple",
    #         "pink",
    #         "black",
    #     ]
    # )
    # if dataset == "CIFAR10":
    #     for i, color in zip(range(n_classes), colors):
    #         plt.plot(
    #             recall[i],
    #             precision[i],
    #             color=color,
    #             lw=2,
    #             label="Class {0} (area = {1:0.2f})" "".format(i, average_precision[i]),
    #         )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-averaged Precision-Recall curve - FEMNIST")
    plt.legend(loc="lower right")
    # Get current time
    now = datetime.now()
    # Format as string (YYYYMMDD_HHMMSS format)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(f"{ROOT_DIR}/split/plots/"):
        os.makedirs(f"{ROOT_DIR}/split/plots/")
    plt.savefig(
        f"{ROOT_DIR}/FL/plots/precision_recall_curve_FEMNIST_{timestamp_str}.png"
    )
    if wandb_track:
        wandb.log({"precision_recall_curve": plt})
    if showPlot:
        plt.show()


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, batch_size, num_epochs):
        # super().__init__()
        self.net = models.CNNFemnist().to(DEVICE)
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def load_dataset(self, train_path, test_path):
        self.trainloader = torch.load(train_path)
        self.testloader = torch.load(test_path)

        # Calculate the total number of samples
        num_trainset = len(self.trainloader) * self.batch_size
        num_testset = len(self.testloader) * self.batch_size
        self.num_examples = {"testset": num_trainset, "trainset": num_testset}
        self.true_test_labels = self.testloader.dataset.targets

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
        loss, accuracy, f1, precision, recall, scores, labels = test(
            self.net, self.testloader
        )

        plot_precision_recall(labels, scores, self.true_test_labels)

        return (
            float(loss),
            self.num_examples["testset"],
            {
                "accuracy": float(accuracy),
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "loss": loss,
                "predictions": scores,
                "true_labels": labels,
            },
        )

    def start_client(self):
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=self)


if __name__ == "__main__":
    main()
