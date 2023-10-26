import argparse
import os
import classification
import regression
import centralized
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


def main():
    args = parse_arguments()
    actual_config = config.Config(args.config)

    centralized.main(actual_config, args.task)

    """
    if args.task == "classification":

            classification.main(actual_config)
        elif args.task == "regression":
            regression.main(actual_config)
        """


if __name__ == "__main__":
    main()
