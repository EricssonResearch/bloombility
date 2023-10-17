#!/usr/bin/env python

import flwr as fl
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

# navigate to the root of the project and import the bloom package
import bloom
from bloom import models

# PARAMS
# Number of rounds of federated learning
n_rounds = 3
# Strategy  ["FedAvg", "FedAdam"]
strat = "FedAvg"


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


# Create an instance of the model and get the parameters
params = get_parameters(models.FedAvgCNN())


def weighted_average(metrics):
    acc = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(acc) / sum(examples)}


# Pass parameters to the Strategy for server-side parameter initialization
strategy = None
if strat == "FedAdam":
    strategy = fl.server.strategy.FedAdam(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=fl.common.ndarrays_to_parameters(params),
        evaluate_metrics_aggregation_fn=weighted_average,
        eta=0.01,
        beta_1=0.9,
        eta_l=0.1,
    )
elif strat == "FedAvg":
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average
    )

print("strategy: ", strategy)
print("Setting up flower server...")
fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=n_rounds), strategy=strategy
)
