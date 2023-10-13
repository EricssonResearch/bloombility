import wandb  # for tracking experiments
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from bloom import models
from bloom import load_data
from bloom import config


def wandb_setup(
    wandb_key: str, _dataset: str, _opt: str, _loss: str, hyper_params: dict
) -> None:
    """logs in to wandb and sets up experiment metadata.

        The metadata can be accessed via "files" tab of an experiment in the website
        and is used to identify the configuration of an experiment even after it has ended.

    Params:
        wandb_key: login key
        _dataset: which dataset the model is trained on
        _opt: which optimizer is used
        _loss: which loss function is used
        hyper_params: configuration dictionary including learning rate and number of epochs
    """
    wandb.login(anonymous="never", key=wandb_key)
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        entity="cs_team_b",
        project="bloomnet_visualization",
        # track hyperparameters and run metadata
        config={
            "learning_rate": hyper_params["learning_rate"],
            "dataset": _dataset,
            "optimizer": _opt,
            "epochs": hyper_params["num_epochs"],
            "loss": _loss,
        },
    )
    return


def setup_config(config: config.Config, task: str) -> None:
    """
    reads config, downloads dataset, preprocesses it,
    defines the chosen model, optimizer and loss, and starts training
    """
    _dataset = config.get_chosen_datasets("regression")
    _opt = config.get_chosen_optimizers("regression")
    _loss = config.get_chosen_loss("regression")
    wandb_track = config.get_wand_active()
    wandb_key = config.get_wandb_key()
    hyper_params = config.get_hyperparams()

    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Task:", task)
    print("Device:", device)
    print("Dataset: ", _dataset)
    print("Optimizer: ", _opt)
    print("Loss: ", _loss)
    print("Hyper-parameters: ", hyper_params)

    if wandb_track:
        wandb_setup(wandb_key, _dataset, _opt, _loss, hyper_params)

    return


def main(config: config.Config) -> None:
    pass


if __name__ == "__main__":
    main()
