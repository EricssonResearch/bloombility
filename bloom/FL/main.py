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
    # PARAMS
    # Number of rounds of federated learning
    n_rounds = cfg.server.num_rounds

    # Strategies available:  ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]
    strategy = cfg.server.strategy
    # wandb experiments
    wandb_track = bool(cfg.main.wandb_active)
    wandb_key = cfg.main.wandb_key

    # Strategies available:  ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]
    batch_size = cfg.client.hyper_params.batch_size
    num_epochs = cfg.client.hyper_params.num_epochs

    num_clients = cfg.main.num_clients

    DATA_DISTRIBUTOR(num_clients)

    subprocess.run(["chmod", "+x", "server/server.py"], check=True)

    subprocess.Popen(
        [
            os.path.join(server_path, "server.py"),
            f"{n_rounds}",
            f"{strategy}",
            f"{wandb_track}",
            f"{wandb_key}",
            f"{num_clients}",
        ]
    )

    subprocess.run(["chmod", "+x", "client/client.py"], check=True)
    for i in range(1, num_clients + 1):
        trainloader_str = (
            f"{ROOT_DIR}/load_data/datasets/train_dataset{i}_{num_clients}.pth"
        )
        testloader_str = f"{ROOT_DIR}/load_data/datasets/test_dataset.pth"
        subprocess.Popen(
            [
                os.path.join(client_path, "client.py"),
                f"{batch_size}",
                f"{num_epochs}",
                trainloader_str,
                testloader_str,
            ]
        )


if __name__ == "__main__":
    main()
