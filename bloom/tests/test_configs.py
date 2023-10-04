import os
import sys
import yaml

default_config = "default_config.yaml"

# ============================== read yaml ==============================================


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


def get_available_loss(config):
    available_loss = []
    chosen_task = config["task"]["chosen"]
    if chosen_task == "regression":
        available_loss = config["loss_functions"]["regression"]["available"]
    else:
        available_loss = config["loss_functions"]["classification"]["available"]
    return available_loss


def get_chosen_loss(config):
    chosen_task = config["task"]["chosen"]
    if chosen_task == "regression":
        return config["loss_functions"]["regression"]["chosen"]
    else:
        return config["loss_functions"]["classification"]["chosen"]


def get_available_datasets(config):
    return config["datasets"]["available"]


def get_chosen_datasets(config):
    return config["datasets"]["chosen"]


def get_available_tasks(config):
    return config["task"]["available"]


def get_chosen_task(config):
    return config["task"]["chosen"]


def get_wandb_active(config):
    return config["wandb"]["active_tracking"]


def get_wandb_key(config):
    return config["wandb"]["login_key"]


# ============================== asserts ==============================================


def check_chosen_in_available(config):
    assert get_chosen_task(config) in get_available_tasks(
        config
    ), "chosen task is not available"
    assert get_chosen_datasets(config) in get_available_datasets(
        config
    ), "chosen dataset is not available"
    assert get_chosen_loss(config) in get_available_loss(
        config
    ), "chosen loss is not available"


def wandb_config(config):
    wandb_key = get_wandb_key(config)
    assert wandb_key is not None, "put wandb API key in yaml file if you want tracking"
    assert isinstance(
        wandb_key, str
    ), "wrong format of wandb API key in yaml file: should be str"
    assert (
        len(wandb_key) == 40
    ), "wrong format of wandb API key in yaml file: should be 40 chars long"


def main():
    config_file_location = os.path.join(
        os.path.dirname(__file__), "..", "central", default_config
    )
    config = read_config_file(config_file_location)

    if get_wandb_active(config):
        wandb_config(config)

    check_chosen_in_available(config)


if __name__ == "__main__":
    main()
