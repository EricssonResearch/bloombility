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

    data_distributor = DATA_DISTRIBUTOR(num_clients)
    trainloaders = data_distributor.get_trainloaders()
    testloader = data_distributor.get_testloader()

    client_fn = generate_client_fn(trainloaders, testloader)

    server = FlowerServer(strategy=strategy_str, num_rounds=n_rounds)
    server.start_simulation(client_fn, num_clients)


if __name__ == "__main__":
    main()
