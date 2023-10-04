import argparse
import os
import simple_classification as classification
import simple_regression as regression

default_config = os.path.join(os.path.dirname(__file__), "default_config.yaml")


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
        "-c", "--config", default=default_config, required=False, dest="config"
    )

    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.task == "classification":
        classification.main(args.config)
    elif args.task == "regression":
        regression.main(args.config)


if __name__ == "__main__":
    main()
