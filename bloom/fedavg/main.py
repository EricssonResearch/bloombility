from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
from client import generate_client_fn


def main():
    num_clients = 5
    data_distributor = DATA_DISTRIBUTOR(num_clients)
    trainloaders = data_distributor.get_trainloaders()
    # testloader = data_distributor.get_testloader()

    client_fn = generate_client_fn(trainloaders)
    return client_fn


if __name__ == "__main__":
    main()
