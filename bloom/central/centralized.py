import wandb  # for tracking experiments
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from bloom import models
from bloom import load_data
from bloom import config

import classification as clas
import regression as regr


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


def setup_config(
    config: config.Config, task: str
) -> tuple[str, str, str, dict, str, bool]:
    """
    reads config and starts wandb if tracking is activated

    Params:
        config: config class that mirrors the config yaml file
        task: the chosen learning task, either classification or regression
    Returns:
        _opt: the optimizer to update the model with
        _loss: the loss function to calculate the difference between expected and actual result
        hyper_params: yaml config dictionary with model hyperparameters
        device: where the calculations are performed, cuda (gpu) or cpu
        wandb_track: whether or not to track and visualize experiment performance with wandb

    """
    _dataset = config.get_chosen_datasets(task)
    _opt = config.get_chosen_optimizers(task)
    _loss = config.get_chosen_loss(task)
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

    return _dataset, _opt, _loss, hyper_params, device, wandb_track


def training(
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    model: nn.Module,
    num_epochs: int,
    optimizer,
    cost,
    wandb_track: bool,
    task: str,
    device: str,
) -> None:
    """
    trains the model on training dataset

    for each epoch, do the following:
        present each image from the training dataset to the model and save its output label.
        With the loss funct, calculate how far off the expected result is from the actual result.
        Propagate the difference of model weights backwards through the model to improve classification.
        Repeat this procedure for each image in the training dataset.

    Afterwards, call classification_accuracy to estimate model performance on test set.

    Params:
        trainloader: the preprocessed training set in a lightweight format
        testloader: the preprocessed testing set in a lightweight format
        model: the NN model to be trained
        num_epochs: number of epochs to train the model
        optimizer: the optimizer to update the model with
        cost: the loss function to calculate the difference between expected and actual result
        wandb_track: whether or not to log experiment performance to wandb
        task: learning task to be performed (classification / regression)
        device: where the calculations are performed, cuda (gpu) or cpu

    """
    # this is defined to print how many steps are remaining when training
    total_step = len(trainloader)

    for epoch in range(num_epochs):
        # set to training mode
        model.train()
        epoch_loss = 0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)
            epoch_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if wandb_track:
                # log metrics to wandb
                wandb.log({"step_loss": loss.item()})

            # print progress to console
            if (i + 1) % 400 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

        # set to evaluation mode
        model.eval()

        # task-based accuracy calculation
        if task == "classification":
            acc_per_epoch = clas.classification_accuracy(testloader, model, device)
        elif task == "regression":
            acc_per_epoch = regr.regression_accuracy(testloader, model, device, 0.10)
        else:
            print("unrecognized task!")
            quit()

        print("epoch accuracy: ", acc_per_epoch)
        if wandb_track:
            wandb.log(
                {
                    "epoch_loss": epoch_loss / len(trainloader),
                    "epoch_acc": acc_per_epoch,
                }
            )

    if wandb_track:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()


def main(config: config.Config, task: str) -> None:
    """based on config and chosen task, sets up dataset, model, loss and optimizer
        and trains and tests model

    Params:
        config: config class that mirrors the config yaml file
        task: the chosen learning task, either classification or regression
    """

    # read config and initiate wand
    _dataset, _opt, _loss, hyper_params, device, wandb_track = setup_config(
        config, task
    )

    # initiate data and config
    if task == "classification":
        trainloader, testloader = clas.get_classification_loaders(
            _dataset, hyper_params
        )
        model = clas.get_classification_model(_dataset, device)
        cost = clas.get_classification_loss(_loss)
        optimizer = clas.get_classification_optimizer(_opt, model, hyper_params)

    elif task == "regression":
        trainloader, testloader = regr.get_regression_loaders(_dataset, hyper_params)
        model = regr.get_regression_model(_dataset, device)
        cost = regr.get_regression_loss(_loss)
        optimizer = regr.get_regression_optimizer(_opt, model, hyper_params)
    else:
        print("unrecognized task")
        quit()

    # start training and testing
    training(
        trainloader,
        testloader,
        model,
        hyper_params["num_epochs"],
        optimizer,
        cost,
        wandb_track,
        task,
        device,
    )


if __name__ == "__main__":
    main()
