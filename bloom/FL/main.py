import wandb

from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
from client import generate_client_fn
from server import FlowerServer
from bloom import ROOT_DIR

import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

config_path = os.path.join(ROOT_DIR, "config", "federated")


@hydra.main(config_path=config_path, config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(config_path)
    print(OmegaConf.to_yaml(cfg))
    # PARAMS
    # Number of rounds of federated learning
    n_rounds = cfg.server.num_rounds

    # Strategies available:  ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]
    strategy = cfg.server.strategy
    batch_size = cfg.client.hyper_params.batch_size
    num_epochs = cfg.client.hyper_params.num_epochs

    num_clients = cfg.main.num_clients

    data_split = cfg.main.data_split_chosen
    data_split_config = cfg.main.data_split_config

    data_distributor = DATA_DISTRIBUTOR(num_clients, data_split_config, data_split)
    # wandb experiments
    wandb_track = cfg.main.wandb_active
    wandb_key = cfg.main.wandb_key

    if wandb_track and wandb.run is None:
        wandb.login(anonymous="never", key=wandb_key)
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            entity="cs_team_b",
            project="non_iid_client_reporting_fn",
            # track hyperparameters and run metadata
            config={
                "method": "federated",
                "n_rounds": n_rounds,
                "strategy": strategy,
                "clients": num_clients,
            },
        )

    trainloaders = data_distributor.get_trainloaders()
    print("Amount of loaders:", len(trainloaders))

    for trainloader in trainloaders:
        print("Len of loader: ", len(trainloader) * batch_size)
        if wandb_track:
            wandb.log({"trainloader_len": len(trainloader) * batch_size})

    testloader = data_distributor.get_testloader()

    client_fn = generate_client_fn(trainloaders, testloader, batch_size, num_epochs)

    server = FlowerServer(strategy=strategy, num_rounds=n_rounds)
    server.start_simulation(
        client_fn, cfg.main.num_clients, cfg.client.num_cpu, cfg.client.num_gpu
    )

    if wandb_track:
        wandb.finish()


if __name__ == "__main__":
    main()
