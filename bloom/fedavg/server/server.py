#!/usr/bin/env python

import flwr as fl
from typing import List
import numpy as np
from utils import get_parameters, define_strategy, weighted_average

from bloom import models


class FlowerServer:
    def __init__(self, strategy: str, num_rounds: int) -> None:
        # Create an instance of the model and get the parameters
        self.params = get_parameters(models.FedAvgCNN())
        # Pass parameters to the Strategy for server-side parameter initialization
        self.strategy = define_strategy(strategy, self.params)
        self.num_rounds = num_rounds

        print("strategy: ", strategy)
        print("Setting up flower server...")

    def get_params(self):
        return self.params

    def get_strategy(self):
        return self.strategy

    def start_simulation(self, client_fn, num_clients):
        history = fl.simulation.start_simulation(
            client_fn=client_fn,  # a function that spawns a particular client
            num_clients=num_clients,  # total number of clients
            config=fl.server.ServerConfig(
                num_rounds=self.num_rounds
            ),  # minimal config for the server loop telling the number of rounds in FL
            strategy=self.strategy,  # our strategy of choice
            client_resources={
                "num_cpus": 2,
                "num_gpus": 0.0,
            },  # (optional) controls the degree of parallelism of your simulation.
            # Lower resources per client allow for more clients to run concurrently
            # (but need to be set taking into account the compute/memory footprint of your workload)
            # `num_cpus` is an absolute number (integer) indicating the number of threads a client should be allocated
            # `num_gpus` is a ratio indicating the portion of gpu memory that a client needs.
        )

        return history
