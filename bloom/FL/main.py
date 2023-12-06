from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
from bloom import ROOT_DIR

import os
import yaml
import argparse

CONFIG_PATH = os.path.join(ROOT_DIR, "config", "federated")


def main():
    parser = argparse.ArgumentParser(prog="main.py")

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
    dataset = cfg["main"]["dataset"]

    DATA_DISTRIBUTOR(num_clients, dataset)


if __name__ == "__main__":
    main()
