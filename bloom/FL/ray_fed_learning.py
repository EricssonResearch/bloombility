from copy import deepcopy
from typing import List
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn import Module
from torch.optim import SGD
import torch.nn as nn
import ray


class Model(nn.Module):
    def __init__(self, input_layer_size, num_labels=10):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_layer_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_labels),
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output


def loss(model: Module, dataset, loss_fn) -> float:
    """calculates the loss of a given dataset and model

    Args:
        model (Module): the model
        dataset (_type_): the dataset
        loss_f (_type_): the loss function

    Returns:
        float: the average loss
    """
    loss = 0.0
    for features, labels in dataset:
        predictions = model(features)
        loss += loss_fn(predictions, labels)

    loss /= len(dataset)
    return loss


def accuracy(model: Module, dataset) -> float:
    """calculats the accuracy for a given dataset and model

    Args:
        model (Module): the model
        dataset (_type_): the dataset

    Returns:
        float: the accuracy
    """
    correct = 0.0
    total = 0.0
    for features, labels in iter(dataset):
        predictions = model(features)
        _, predicted = predictions.max(1, keepdim=True)
        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()
        total += len(predicted)

    accuracy = 100 * correct / total
    return accuracy


def train_step(model: Module, optimizer, train_data, loss_f) -> float:
    """performs one training step - this includes iterating
        over all features/labels in the train data set
        and measuring average loss

    Args:
        model (_type_): the model to be trained
        optimizer (_type_): optimizer used for training
        train_data (_type_): train data
        loss_f (_type_): loss function

    Returns:
        (float): average loss
    """

    total_loss = 0
    for features, labels in train_data:
        # forward propagation
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_f(predictions, labels)
        total_loss += loss.item()

        # backward propagation
        loss.backward()
        optimizer.step()

    return total_loss / len(train_data)


train_step_ray = ray.remote(train_step)
loss_ray = ray.remote(loss)


def local_learning(
    model: Module, optimizer, train_data, test_data, epochs: int, loss_f
) -> (List[float], List[float]):
    """perform the local training which means using a specific (per worker)
        train dataset and test dataset which is trained using an optimizer
        and a loss function for a number of epoches

        as a result we record the loss per epoch for the train and test
        data set

    Args:
        model (_type_): model to be trained
        List (_type_): optimizer

    Returns:
        _type_: _description_
    """

    train_loss_per_epoch = []
    test_loss_per_epoch = []

    for _ in range(epochs):
        # training
        model.train()
        train_loss = train_step_ray.remote(
            model=model, optimizer=optimizer, train_data=train_data, loss_f=loss_f
        )
        train_loss_per_epoch.append(train_loss)

        # evaluate
        model.eval()
        test_loss = loss_ray.remote(model=model, dataset=test_data, loss_fn=loss_f)
        test_loss_per_epoch.append(test_loss)

    return train_loss_per_epoch, test_loss_per_epoch, model


def set_to_zero(model: Module):
    """sets all parameters of every layer to zero
        by subtracting them by itself

    Args:
        model (_type_): the model to receive zero parameters
    """
    for layer_weights in model.parameters():
        layer_weights.data.sub_(layer_weights.data)


def params_from_model(model: Module) -> List:
    params = list(model.parameters())
    return [tensors.detach() for tensors in params]


def average(model: Module, client_params, weights: List[float]):
    """perform federated averaging for the parameters received
    given their weight

    Args:
        model (_type_): model
        client_params (_type_): list of parameters per worker
        weights (_type_): list of weights

    Returns:
        _type_: a new model which is the fedavg of model
    """
    # we obtain a new model
    new_model = deepcopy(model)

    # we set all neural parameters to zero
    set_to_zero(new_model)

    # for every participant
    for i, client_param in enumerate(client_params):
        # for every layer of each participant
        for idx, layer_weights in enumerate(new_model.parameters()):
            # calculate the contribution
            contribution = client_param[idx].data * weights[i]
            # add back to the new model
            layer_weights.data.add_(contribution)

    # return the new model
    return new_model


local_learning_ray = ray.remote(local_learning)


def fed_avg(
    model: Module,
    loss_fns,
    training_sets,
    testing_sets,
    epochs: int,
    learning_rate: float,
    current_iteration: int,
):
    """performs one iteration of federated averaging sequentially for every training and test set

    Args:
        model (_type_): model to be federated
        loss_fns (_type_): list of loss functions
        training_sets (_type_): training sets
        testing_sets (_type_): test sets
        epochs (int): epochs
        learning_rate (float): learning rate
        current_iteration (int): iteration

    Returns:
        _type_: the averaged model
    """

    train_server_loss = []
    test_server_loss = []

    assert len(training_sets) == len(testing_sets)
    assert len(loss_fns) == len(training_sets)

    # identify the contribution of each participant
    n_samples = sum([len(train_dataset) for train_dataset in training_sets])
    weights = [len(train_dataset) / n_samples for train_dataset in training_sets]

    clients_params = []
    clients_losses = {}

    # every worker receives the same initial model
    # that model is "cloned" and trained locally
    # then the results are saved in array to be used for
    # federated averaging
    for i, training_set in enumerate(training_sets):
        local_model = deepcopy(model)
        local_model.train()

        local_optimizer = SGD(local_model.parameters(), lr=learning_rate)
        # train_loss_per_epoch, test_loss_per_epoch = local_learning(model=local_model,
        #                                                            optimizer=local_optimizer,
        #                                                            train_data=training_set,
        #                                                            test_data=testing_sets[i],
        #                                                            epochs=epochs,
        #                                                            loss_f=loss_fns[i])

        future = local_learning_ray.remote(
            model=local_model,
            optimizer=local_optimizer,
            train_data=training_set,
            test_data=testing_sets[i],
            epochs=epochs,
            loss_f=loss_fns[i],
        )

        train_loss_per_epoch, test_loss_per_epoch, new_model = ray.get(future)
        print(train_loss_per_epoch, test_loss_per_epoch)

        clients_losses[i] = {
            "train_loss": train_loss_per_epoch,
            "test_loss": test_loss_per_epoch,
        }

        clients_params.append(params_from_model(new_model))

    # fedAvg
    model = average(model, clients_params, weights=weights)

    print(len(clients_params))

    model.eval()

    train_server_loss = 0.0
    for i, train_dl in enumerate(training_sets):
        train_loss = loss(model, train_dl, loss_fns[i])
        train_server_loss += train_loss * weights[i]

    test_server_loss = 0.0
    test_server_acc = 0.0

    for i, test_dl in enumerate(testing_sets):
        test_loss = loss(model, test_dl, loss_fns[i])
        test_acc = accuracy(model, test_dl)
        test_server_acc += test_acc * weights[i]
        test_server_loss += test_loss * weights[i]

    print(
        f"i: {current_iteration} Train server loss: {train_server_loss}"
        f" Test server loss: {test_server_loss} Test server accuracy: {test_server_acc}"
    )

    return model, train_server_loss, test_server_loss, clients_losses


def centralized():
    # basic data transformation for MNIST
    # outputs a 28x28=784 tensors for every sample
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    # downloads the datasets
    train_set = MNIST(
        root="./data", train=True, download=True, transform=data_transform
    )
    test_set = MNIST(
        root="./data", train=False, download=True, transform=data_transform
    )

    BATCH_SIZE = 32

    # data loader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    model = Model(input_layer_size=784)

    EPOCHS = 10

    LEARNING_RATE = 0.01

    loss_fns = [CrossEntropyLoss()]
    train_dls = [train_loader]
    test_dls = [test_loader]

    train_server_losses = []
    test_server_losses = []
    client_losses = []

    # centralized training
    model, train_server_loss, test_server_loss, client_loss = fed_avg(
        model=model,
        loss_fns=loss_fns,
        training_sets=train_dls,
        testing_sets=test_dls,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        current_iteration=iteration,
    )

    train_server_losses.append(train_server_loss)
    test_server_losses.append(test_server_loss)
    client_losses.append(client_loss)


if __name__ == "__main__":
    ray.init()
    # federated learning in iid setting

    # basic data transformation for MNIST
    # outputs a 28x28=784 tensors for every sample
    data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    # downloads the datasets
    train_set = MNIST(
        root="./data", train=True, download=True, transform=data_transform
    )
    test_set = MNIST(
        root="./data", train=False, download=True, transform=data_transform
    )

    BATCH_SIZE = 32

    # data loader
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    model = Model(input_layer_size=784)

    N_ITER = 10

    EPOCHS = 1

    LEARNING_RATE = 0.01

    # split data in two
    train_set = []
    for train_features, train_labels in train_loader:
        train_set.append((train_features, train_labels))

    test_set = []
    for test_features, test_labels in test_loader:
        test_set.append((test_features, test_labels))

    train_split = int(len(train_set) / 2)
    test_split = int(len(test_set) / 2)

    train_dls = [train_set[0:train_split], train_set[train_split:]]
    test_dls = [test_set[0:test_split], test_set[test_split:]]

    loss_fns = [CrossEntropyLoss(), CrossEntropyLoss()]

    train_server_losses = []
    test_server_losses = []
    client_losses = []

    for iteration in range(N_ITER):
        model, train_server_loss, test_server_loss, client_loss = fed_avg(
            model=model,
            loss_fns=loss_fns,
            training_sets=train_dls,
            testing_sets=test_dls,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            current_iteration=iteration,
        )

        train_server_losses.append(train_server_loss)
        test_server_losses.append(test_server_loss)
        client_losses.append(client_loss)

    ray.shutdown()
