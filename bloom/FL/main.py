from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
from bloom import ROOT_DIR

import os
import argparse

config_path = os.path.join(ROOT_DIR, "config", "federated")
server_path = os.path.join(ROOT_DIR, "FL", "server")
client_path = os.path.join(ROOT_DIR, "FL", "client")


def main():
    parser = argparse.ArgumentParser(prog="main.py")

    parser.add_argument("-n", "--num_clients", type=int, default=4, dest="num_clients")
    args = parser.parse_args()

    num_clients = args.num_clients

    DATA_DISTRIBUTOR(num_clients)


if __name__ == "__main__":
    main()
