#!/usr/bin/env python

import flwr as fl
from typing import List
import numpy as np
from bloom.FL.server.utils import get_parameters, define_strategy
import wandb
from bloom import models
from bloom import ROOT_DIR
import sys
import logging


def main():
    # PARAMS
    # Number of rounds of federated learning
    n_rounds = int(sys.argv[1])

    # Strategies available:  ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]
    strategy = sys.argv[2]
    # wandb experiments
    wandb_track = False
    if sys.argv[3] == "True":
        wandb_track = True

    wandb_key = sys.argv[4]
    num_clients = int(sys.argv[5])

    if wandb_track:
        wandb.login(anonymous="never", key=wandb_key)
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            entity="cs_team_b",
            project="bloomnet_visualization",
            # track hyperparameters and run metadata
            config={
                "method": "federated",
                "n_rounds": n_rounds,
                "strategy": strategy,
                "clients": num_clients,
            },
        )

    server = FlowerServer(
        strategy=strategy, num_rounds=n_rounds, wandb_track=wandb_track
    )
    server.start_server()
    sys.stdout.flush()

    if wandb_track:
        wandb.finish()


class FlowerServer:
    def __init__(self, strategy: str, num_rounds: int, wandb_track: bool) -> None:
        # Create an instance of the model and get the parameters
        self.params = get_parameters(models.FedAvgCNN())
        # Pass parameters to the Strategy for server-side parameter initialization
        self.strategy = define_strategy(strategy, wandb_track, self.params)
        self.num_rounds = num_rounds

        print("strategy: ", strategy)

    def get_params(self):
        return self.params

    def get_strategy(self):
        return self.strategy

    def start_server(self):
        print("Setting up flower server...")
        fl.server.start_server(
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=self.strategy,
        )


if __name__ == "__main__":
    main()
