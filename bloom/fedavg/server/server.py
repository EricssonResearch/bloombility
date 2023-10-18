#!/usr/bin/env python

import flwr as fl
from typing import List
import numpy as np
from utils import get_parameters, get_strategy, weighted_average

from bloom import models

# PARAMS
# Number of rounds of federated learning
n_rounds = 3

# Strategies available:  ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]
strat = "FedAvg"


# Create an instance of the model and get the parameters
params = get_parameters(models.FedAvgCNN())

# Pass parameters to the Strategy for server-side parameter initialization
strategy = None
strategy = get_strategy(strat, params)

print("strategy: ", strategy)
print("Setting up flower server...")
fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=n_rounds), strategy=strategy
)
