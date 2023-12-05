#!/usr/bin/env python

import flwr as fl
from bloom.FL.server.utils import get_parameters, define_strategy
import wandb
from bloom import models
from bloom import ROOT_DIR
import argparse
import logging
import os
import yaml

CONFIG_PATH = os.path.join(ROOT_DIR, "config", "federated")


def main():
    parser = argparse.ArgumentParser(
        prog="server.py", description="runs a flower server"
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
    args = parser.parse_args()

    cfg_path = os.path.join(CONFIG_PATH, args.config_file)
    with open(cfg_path, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    num_clients = args.num_clients

    # Configure logging in each subprocess
    logging.basicConfig(filename="server.log", level=logging.INFO)

    # Example log statement with explicit flushing
    logging.debug("Debug message")
    logging.getLogger().handlers[0].flush()

    # Number of rounds of federated learning
    n_rounds = cfg["server"]["num_rounds"]

    # Strategies available:  ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]
    strategy = cfg["server"]["strategy"]
    # wandb experiments
    wandb_track = cfg["server"]["wandb_active"]

    if wandb_track:
        wandb_key = cfg["server"]["wandb_key"]
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
