import ray
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

ray.init()


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


def params_from_model(model: Module) -> List:
    params = list(model.parameters())
    return [tensors.detach() for tensors in params]


def accuracy(model: Module, dataset) -> float:
    correct = 0.0
    total = 0.0
    for features, labels in iter(dataset):
        predictions = model(features)
        _, predicted = predictions.max(1, keepdim=True)
        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()
        total += len(predicted)

    accuracy = 100 * correct / total
    return accuracy


@ray.remote
def remote_train_step(model: Module, optimizer, train_data, loss_f) -> float:
    total_loss = 0
    for features, labels in train_data:
        optimizer.zero_grad()
        predictions = model(features)
        loss = loss_f(predictions, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(train_data)


@ray.remote
def remote_average(model: Module, client_params, weights: List[float]):
    new_model = deepcopy(model)
    for i, client_param in enumerate(client_params):
        for idx, layer_weights in enumerate(new_model.parameters()):
            contribution = client_param[idx].data * weights[i]
            layer_weights.data.add_(contribution)
    return new_model


@ray.remote
def remote_loss(model: Module, dataset, loss_fn) -> float:
    loss = 0.0
    for features, labels in dataset:
        predictions = model(features)
        loss += loss_fn(predictions, labels)
    loss /= len(dataset)
    return loss


@ray.remote
def fed_avg(
    model: Module,
    loss_fns,
    training_sets,
    testing_sets,
    epochs: int,
    learning_rate: float,
    current_iteration: int,
):
    train_server_loss = []
    test_server_loss = []

    assert len(training_sets) == len(testing_sets)
    assert len(loss_fns) == len(training_sets)

    n_samples = sum([len(train_dataset) for train_dataset in training_sets])
    weights = [len(train_dataset) / n_samples for train_dataset in training_sets]

    clients_params = []
    clients_losses = {}

    for i, training_set in enumerate(training_sets):
        local_model = deepcopy(model)
        local_model.train()

        local_optimizer = SGD(local_model.parameters(), lr=learning_rate)
        train_loss_per_epoch_id = remote_train_step.remote(
            local_model, local_optimizer, training_set, loss_fns[i]
        )
        print(train_loss_per_epoch_id)
        clients_losses[i] = {"train_loss": [], "test_loss": []}
        clients_params.append(params_from_model(local_model))

    model = remote_average.remote(model, clients_params, weights=weights)

    # model.eval()
    model_ref = ray.get(model)
    model_ref.eval()

    train_server_loss = 0.0
    for i, train_dl in enumerate(training_sets):
        train_loss = ray.get(remote_loss.remote(model, train_dl, loss_fns[i]))
        train_server_loss += train_loss * weights[i]

    test_server_loss = 0.0
    test_server_acc = 0.0

    for i, test_dl in enumerate(testing_sets):
        test_loss = ray.get(remote_loss.remote(model, test_dl, loss_fns[i]))
        test_acc = accuracy(ray.get(model), test_dl)
        test_server_acc += test_acc * weights[i]
        test_server_loss += test_loss * weights[i]

    print(
        f"i: {current_iteration} Train server loss: {train_server_loss}"
        f" Test server loss: {test_server_loss} Test server accuracy: {test_server_acc}"
    )

    return ray.get(model), train_server_loss, test_server_loss, clients_losses


model = Model(input_layer_size=784)  # initializationmodel
loss_fns = [CrossEntropyLoss(), CrossEntropyLoss()]  # loss function

N_ITER = 10  # number of iterations
N_WORKERS = 4  # number of workers or nodes in Ray cluster
EPOCHS = 10
LEARNING_RATE = 0.01
BATCH_SIZE = 32


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
train_set = MNIST(root="./data", train=True, download=True, transform=data_transform)
test_set = MNIST(root="./data", train=False, download=True, transform=data_transform)

# data loader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

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


# iteration loop to use Ray for parallel execution
for iteration in range(N_ITER):
    result_ids = []
    for i in range(N_WORKERS):
        result_ids.append(
            fed_avg.remote(
                model, loss_fns, train_dls, test_dls, EPOCHS, LEARNING_RATE, iteration
            )
        )
    results = ray.get(result_ids)
    print(results)

ray.shutdown()
