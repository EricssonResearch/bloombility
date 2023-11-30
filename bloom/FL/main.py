from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
from client import FlowerClient
from server import FlowerServer
from bloom import ROOT_DIR

import os
import hydra
import subprocess
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import flwr as fl

config_path = os.path.join(ROOT_DIR, "config", "federated")
server_path = os.path.join(ROOT_DIR, "FL", "server")
client_path = os.path.join(ROOT_DIR, "FL", "client")


@hydra.main(config_path=config_path, config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    num_clients = cfg.main.num_clients

    DATA_DISTRIBUTOR(num_clients)


if __name__ == "__main__":
    main()
