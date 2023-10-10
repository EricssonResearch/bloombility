# Document attribution:
# based on tutorial here: https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

# Current working configuration for simple_regression.py:
#   datasets: CaliforniaHousing
#   optimizers: Adam
#   loss_functions: MSELoss

import wandb  # for tracking experiments
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

from bloom import models

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(config):
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

    print("Device:", device)
    print("Dataset: ", _dataset)
    print("Optimizer: ", _opt)
    print("Loss: ", _loss)
    print("Hyper-parameters: ", hyper_params)

    if wandb_track:
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

    # Read data
    if _dataset == "CaliforniaHousing":
        data = fetch_california_housing()
        X, y = data.data, data.target

    # train-test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, shuffle=True
    )

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

    # Create the model
    model = models.Networks.RegressionModel().to(device)

    # loss function and optimizer
    if _loss == "MSELoss":
        loss_fn = nn.MSELoss()  # mean square error'
    if _opt == "Adam":
        optimizer = optim.Adam(model.parameters(), hyper_params["learning_rate"])

    n_epochs = hyper_params["num_epochs"]  # number of epochs to run
    batch_size = hyper_params["batch_size"]  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_mse = np.inf  # init to infinity
    best_weights = None
    history = []

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=False) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start : start + batch_size].to(device)
                y_batch = y_train[start : start + batch_size].to(device)

                # forward pass
                y_pred = model.forward(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=loss.item(), rmse=np.sqrt(loss.item()))

                if wandb_track:
                    # log metrics to wandb
                    wandb.log({"step mse": loss.item()})
        # evaluate loss at end of each epoch
        with torch.no_grad():
            model.eval()
            y_pred = model.forward(X_test)
            mse = loss_fn(y_pred, y_test)
            # mse = float(mse)
            history.append(mse.item())

            # calc accuracy
            acc_per_epoch = accuracy(X_test, y_test, model, 0.10)

            if wandb_track:
                # log metrics to wandb
                wandb.log({"epoch mse": loss.item(), "epoch accuracy": acc_per_epoch})

            # save best model
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print()
    print("Overall MSE: %.2f" % best_mse)
    print("Overall RMSE: %.2f" % np.sqrt(best_mse))

    if wandb_track:
        # [optional] finish the wandb run, necessary in notebooks
        wandb.finish()


def accuracy(X_test, y_test, model, pct_close):
    """
    evaluates accuracy of network on train dataset

    compares expected with actual output of the model
    when presented with data from previously unseen testing set.
    The pct_close is an allowed "error range" within which the prediction has to be to be considered correct.
    This ensures that the model does not just "know the training data results by heart",
    but has actually found and learned patterns in the training data

    Args:
        testloader: the preprocessed testing set in a lightweight format
        model: the pretrained(!) NN model to be evaluated

    """
    n_correct = 0
    n_wrong = 0

    with torch.no_grad():
        for image, label in zip(X_test, y_test):
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            if torch.abs(output - label) < torch.abs(pct_close * label):
                n_correct += 1
            else:
                n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc


# call main function when running the script
if __name__ == "__main__":
    main()
