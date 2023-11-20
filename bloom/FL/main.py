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


@hydra.main(config_path=config_path, config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    # PARAMS
    # Number of rounds of federated learning

    # Strategies available:  ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]
    batch_size = cfg.client.hyper_params.batch_size
    num_epochs = cfg.client.hyper_params.num_epochs

    num_clients = cfg.main.num_clients

    data_distributor = DATA_DISTRIBUTOR(num_clients)

    trainloaders = data_distributor.get_trainloaders()
    testloader = data_distributor.get_testloader()

    subprocess.Popen(["./server/server.py"])

    for i in range(num_clients):
        client = FlowerClient(trainloaders, testloader, batch_size, num_epochs)
        fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
