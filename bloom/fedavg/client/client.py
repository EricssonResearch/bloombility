#!/usr/bin/env python3

import flwr as fl
import torch

from centralized import load_data
from centralized import train
from centralized import test
from centralized import Net
from collections import OrderedDict


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FlowerClient(fl.client.NumPyClient):
    # return the model weight as a list of NumPy ndarrays
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    # update the local model weights with the parameters received from the server
    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


# Load model and data
print("Setting up local model...")
net = Net().to(DEVICE)
print("Fetching local data...")
trainloader, testloader, num_examples = load_data()

print("Local data size:", num_examples)

# Boot up client
print("Connecting to flower server...")
fl.client.start_numpy_client(server_address="flower_server:8080", client=FlowerClient())
