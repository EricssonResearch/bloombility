import wandb

from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
from client import generate_client_fn
from server import FlowerServer


def main():
    # PARAMS
    # Number of rounds of federated learning
    n_rounds = 3

    # Strategies available:  ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]
    strategy_str = "FedAvg"
    num_clients = 2

    # wandb experiments
    wandb_track = True
    wandb_key = "<your key here>"

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
                "strategy": strategy_str,
                "clients": num_clients,
            },
        )

    data_distributor = DATA_DISTRIBUTOR(num_clients)
    trainloaders = data_distributor.get_trainloaders()
    testloader = data_distributor.get_testloader()

    client_fn = generate_client_fn(trainloaders, testloader)

    server = FlowerServer(strategy=strategy_str, num_rounds=n_rounds)
    server.start_simulation(client_fn, num_clients)


if __name__ == "__main__":
    main()
