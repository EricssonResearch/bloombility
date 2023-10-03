# Document attribution:
# based on tutorial here: https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

# Current working configuration for simple_regression.py:
#   datasets: CaliforniaHousing
#   optimizers: Adam
#   loss_functions: MSELoss

import copy
import yaml
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from Networks import RegressionModel


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


def parse_config(config):
    """
    parses the configuration dictionary and returns actual config values

    Args:
        config: config as dictionary object

    """
    chosen_task = config["task"]["chosen"]
    if chosen_task == "regression":
        chosen_loss = config["loss_functions"]["regression"]["chosen"]
    else:
        chosen_loss = config["loss_functions"]["classification"]["chosen"]
    return (
        config["datasets"]["chosen"],
        config["optimizers"]["chosen"],
        chosen_loss,
        config["hyper-params"],
    )


def main():
    """
    reads config, downloads dataset, preprocesses it,
    defines the chosen model, optimizer and loss, and starts training
    """
    config_file = sys.argv[1]
    # config_file = os.path.join(os.getcwd(), 'assignments', 'centralized_CNN', 'config.yaml')
    config = read_config_file(config_file)
    _dataset, _opt, _loss, hyper_params = parse_config(config)
    print("Dataset: ", _dataset)
    print("Optimizer: ", _opt)
    print("Loss: ", _loss)
    print("Hyper-parameters: ", hyper_params)

    # Read data
    if _dataset == "CaliforniaHousing":
        data = fetch_california_housing()
        X, y = data.data, data.target

    # train-test split for model evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, shuffle=True
    )

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    # Create the model
    model = RegressionModel()

    # loss function and optimizer
    if _loss == "MSELoss":
        loss_fn = nn.MSELoss()  # mean square error'
        print("Loss function: ", loss_fn)
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
                X_batch = X_train[start : start + batch_size]
                y_batch = y_train[start : start + batch_size]
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
        # evaluate accuracy at end of each epoch
        with torch.no_grad():
            model.eval()
            y_pred = model.forward(X_test)
            mse = loss_fn(y_pred, y_test)
            # mse = float(mse)
            history.append(mse.item())

            # save best model
            if mse < best_mse:
                best_mse = mse
                best_weights = copy.deepcopy(model.state_dict())

    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training history")
    plt.show()


# call main function when running the script
if __name__ == "__main__":
    main()
