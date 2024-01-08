from mpi4py import MPI
from copy import deepcopy
from typing import List

import torch
from torch import nn
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


class HeadModel(nn.Module):
    def __init__(self, input_layer_size=32, num_labels=10):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(nn.Linear(input_layer_size, num_labels))

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output


class WorkerModel(nn.Module):
    def __init__(self, input_layer_size):
        super().__init__()
        cut_layer_size = 32
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_layer_size, cut_layer_size), nn.ReLU(), nn.Dropout(0.2)
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output


def get_mnist(batch_size: int):
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

    # data loader
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def accuracy(model: Module, head_model: Module, test_loader: DataLoader):
    model.eval()
    head_model.eval()

    correct_test = 0
    total_test_labels = 0
    for input_data, labels in test_loader:
        split_layer_tensor = model(input_data)
        logits = head_model(split_layer_tensor)

        _, predictions = logits.max(1)

        correct_test += predictions.eq(labels).sum().item()
        total_test_labels += len(labels)

    test_acc = correct_test / total_test_labels
    return test_acc


def train_head_step(head_model: Module, input_tensor, labels, loss_fn, head_optimizer):
    # forward propagation on the head model
    head_optimizer.zero_grad()
    logits = head_model(input_tensor)
    loss = loss_fn(logits, labels)
    # backward propagation on the head model
    loss.backward(retain_graph=True)
    head_optimizer.step()

    # output to be used to do backward propagation on the worker model
    return input_tensor.grad, loss


def split_train_step(
    worker_model: Module,
    head_model: Module,
    loss_fn,
    worker_optimizer,
    head_optimizer,
    train_data: DataLoader,
    test_data: DataLoader,
    comm,
):
    total_loss = 0

    worker_model.train()
    head_model.train()

    for features, labels in train_data:
        # forward propagation worker model
        worker_optimizer.zero_grad()

        cut_layer_tensor = worker_model(features)

        client_output = cut_layer_tensor.clone().detach().requires_grad_(True)

        # perform forward propagation on the head model and then we receive
        # the gradients to do backward propagation
        grads, loss = train_head_step(
            head_model=head_model,
            input_tensor=client_output,
            labels=labels,
            loss_fn=loss_fn,
            head_optimizer=head_optimizer,
        )

        # backward propagation on the worker node
        cut_layer_tensor.backward(grads)
        worker_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_data)


def split_nn(
    worker_models: List[Module],
    head_model: Module,
    head_loss_fn,
    training_sets: List[DataLoader],
    testing_sets: List[DataLoader],
    epochs: int,
    learning_rate: float,
    comm,
):
    assert len(worker_models) == len(training_sets)

    optimizers = []

    for worker_model in worker_models:
        optimizers.append(SGD(worker_model.parameters(), lr=learning_rate))

    head_optimizer = SGD(head_model.parameters(), lr=learning_rate)

    history = {}

    for i in range(len(worker_models)):
        history[i] = {"train_acc": [], "test_acc": [], "train_loss": []}

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank != 0:
        for e in range(epochs):
            train_loss = split_train_step(
                worker_model=worker_models[rank - 1],
                head_model=head_model,
                loss_fn=head_loss_fn,
                worker_optimizer=optimizers[rank - 1],
                head_optimizer=head_optimizer,
                train_data=training_sets[rank - 1],
                test_data=testing_sets[rank - 1],
                comm=comm,
            )

            history[rank - 1]["train_loss"].append(train_loss)
            print(f"Worker {i} Epoch {e} - Training loss: {train_loss}")
        comm.send(history, dest=0)
    else:
        for pid in range(1, size):
            history = comm.recv(source=pid)
            # Todo : log details

    if rank != 0:
        print(
            f"Worker {rank} acc: {accuracy(worker_models[rank-1], head_model, testing_sets[rank-1])}"
        )

    return history


def split_learning():
    # Init MPI
    comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    size = comm.Get_size()

    # split learning in iid setting
    BATCH_SIZE = 32
    EPOCHS = 5
    LEARNING_RATE = 0.01

    num_clients = size - 1

    train_loader, test_loader = get_mnist(batch_size=BATCH_SIZE)

    worker_models = [WorkerModel(input_layer_size=784) for _ in range(num_clients)]

    head_model = HeadModel()

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

    head_loss_fn = CrossEntropyLoss()

    split_nn(
        worker_models=worker_models,
        head_model=head_model,
        head_loss_fn=head_loss_fn,
        training_sets=train_dls,
        testing_sets=test_dls,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        comm=comm,
    )


if __name__ == "__main__":
    split_learning()
