# Document attribution:
# based on tutorial here: https://machinelearningmastery.com/building-a-regression-model-in-pytorch/

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

from bloom import models


# ----------------------------------------- dataset ------------------------------------------------------
def preprocess_california(
    batch_size: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """preprocess the california dataset into a torch DataLoader

    split data into train and test,
    reshape them into tensors and convert them to DataLoaders

    Params:
        batch_size: size of the batches
        device: where calculations are performed (cuda, which means gpu / cpu)

    Returns:
        trainloader: training data DataLoader object
        testloader: testing data DataLoader object
    """
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

    trainloader = torch.utils.data.DataLoader(
        [[X_train[i], y_train[i]] for i in range(len(y_train))],
        shuffle=True,
        batch_size=batch_size,
    )
    testloader = torch.utils.data.DataLoader(
        [[X_test[i], y_test[i]] for i in range(len(y_test))],
        shuffle=True,
        batch_size=batch_size,
    )

    return trainloader, testloader


def get_regression_loaders(
    _dataset: str, hyper_params: dict
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """based on the chosen dataset, retrieve the data, pre-process it
    and convert it into a DataLoader

    Params:
        _dataset: chosen dataset
        hyper_params: config dictionary with batch size key-value pair

    Returns:
        trainloader: training data DataLoader object
        testloader: testing data DataLoader object
    """
    if _dataset == "CaliforniaHousing":
        trainloader, testloader = preprocess_california(hyper_params["batch_size"])
    else:
        print("Unrecognized dataset")
        quit()
    return trainloader, testloader


# ----------------------------------------- config -------------------------------------------------------
def get_regression_optimizer(
    _opt: str, model: nn.Module, hyper_params: dict
) -> torch.optim:
    """based on yaml config, return optimizer

    Params:
        _opt: chosen optimizer
        model: model to optimize
        hyper_params: yaml config dictionary with at least learning rate defined
    Returns:
        optimizer: configured optimizer
    """
    if _opt == "Adam":
        optimizer = optim.Adam(model.parameters(), hyper_params["learning_rate"])
    else:
        print("Unrecognized optimizer!")
        quit()
    return optimizer


def get_regression_model(_dataset: str, device: str) -> nn.Module:
    """based on the chosen dataset, return correct model
    Params:
        _dataset: chosen dataset
        device: where calculations are performed (cuda, which means gpu / cpu)

    Returns:
        model: model as defined in Networks file
    """
    if _dataset == "CaliforniaHousing":
        model = models.Networks.RegressionModel().to(device)
    else:
        print("Unrecognized dataset")
        quit()
    return model


def get_regression_loss(_loss: str) -> nn:
    """based on the chosen loss, return correct loss function object
    Params:
        _loss: chosen loss

    Returns:
        loss_fn: loss function object
    """
    if _loss == "MSELoss":
        loss_fn = nn.MSELoss()  # mean square error
    else:
        print("Unrecognized loss funct")
        quit()
    return loss_fn


# ------------------------------- accuracy ---------------------------------------------------------------


def regression_accuracy(
    testloader: torch.utils.data.DataLoader,
    model: nn.Module,
    device: str,
    pct_close: float,
) -> float:
    """
    evaluates accuracy of network on test dataset

    compares expected with actual output of the model
    when presented with data from previously unseen testing set.
    The pct_close is an allowed "error range" within which the prediction has to be to be considered correct.
    This ensures that the model does not just "know the training data results by heart",
    but has actually found and learned patterns in the training data

    Params:
        testloader: the preprocessed testing set in a lightweight format
        model: the pretrained(!) NN model to be evaluated
        pct_close: percentage how close the prediction needs to be to be considered correct
    Returns:
        acc: regression accuracy

    """
    n_correct = 0
    n_wrong = 0

    with torch.no_grad():
        # iterate over batches to get outputs per batch
        for i, (image, label) in enumerate(testloader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)

            # for each prediction and each real label within the batch,
            # see if they are close enough
            for out, lab in zip(output, label):
                if torch.abs(out - lab) < torch.abs(pct_close * lab):
                    n_correct += 1
                else:
                    n_wrong += 1

    acc = (n_correct * 1.0) / (n_correct + n_wrong)
    return acc
