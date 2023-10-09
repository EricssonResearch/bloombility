import argparse
import os
import classification
import regression
import yaml

from bloom import config


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run centralized CNN.")

    parser.add_argument(
        "-t",
        "--task",
        default="classification",
        choices=["classification", "regression"],
        required=True,
        dest="task",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=config.Config.DEFAULT_CONFIG,
        required=False,
        dest="config",
    )

    return parser.parse_args()


def read_config_file(config_filepath: str):
    """
    reads the configuration from the YAML file specified
    returns the config as dictionary object

    Args:
        config_filepath: path to the YAML file containing the configuration

    """
    if not (config_filepath.lower().endswith((".yaml", ".yml"))):
        print("Please provide a path to a YAML file.")
        quit()
    with open(config_filepath, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def main():
    args = parse_arguments()
    actual_config = config.Config(args.config)

    if args.task == "classification":
        classification.main(actual_config)
    elif args.task == "regression":
        regression.main(actual_config)


if __name__ == "__main__":
    main()
