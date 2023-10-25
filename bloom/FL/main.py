from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
from client import generate_client_fn
from server import FlowerServer
from bloom import ROOT_DIR

import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

config_path = os.path.join(ROOT_DIR, "config", "federated")


@hydra.main(config_path=config_path, config_name="default_config_FL", version_base=None)
def main(cfg: DictConfig):
    # PARAMS
    # Number of rounds of federated learning
    n_rounds = cfg.server.num_rounds

    # Strategies available:  ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]
    strategy = cfg.server.strategy
    batch_size = cfg.client.hyper_params.batch_size
    num_epochs = cfg.client.hyper_params.num_epochs

    data_distributor = DATA_DISTRIBUTOR(cfg.main.num_clients)
    trainloaders = data_distributor.get_trainloaders()
    testloader = data_distributor.get_testloader()

    client_fn = generate_client_fn(trainloaders, testloader, batch_size, num_epochs)

    server = FlowerServer(strategy=strategy, num_rounds=n_rounds)
    server.start_simulation(client_fn, cfg.main.num_clients)


if __name__ == "__main__":
    main()
