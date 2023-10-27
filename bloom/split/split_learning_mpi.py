from copy import deepcopy
from typing import List
from mpi4py import MPI
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


# Init MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# number of worker models and number of epochs
num_workers = 4
epochs = 5

# worker model for each process
worker_models = [WorkerModel(input_layer_size=784) for _ in range(num_workers)]

# Init a head model
head_model = HeadModel()

learning_rate = 0.01

# Split the data into multiple subsets
train_loader, test_loader = get_mnist(batch_size=32)

train_set = []
for train_features, train_labels in train_loader:
    train_set.append((train_features, train_labels))

test_set = []
for test_features, test_labels in test_loader:
    test_set.append((test_features, test_labels))

train_split = int(len(train_set) / size)
test_split = int(len(test_set) / size)

# train_set = train_set[rank * train_split:(rank + 1) * train_split]
# test_set = test_set[rank * test_split:(rank + 1) * test_split]

# Initialize the head loss function
head_loss_fn = CrossEntropyLoss()

# Create an optimizer for each worker model
optimizers = [SGD(worker.parameters(), lr=learning_rate) for worker in worker_models]

head_optimizer = SGD(head_model.parameters(), lr=learning_rate)

# Training loop
for e in range(epochs):
    for i, worker_model in enumerate(worker_models):
        train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

        train_loss = split_train_step(
            worker_model=worker_model,
            head_model=head_model,
            loss_fn=head_loss_fn,
            worker_optimizer=optimizers[i],
            head_optimizer=head_optimizer,
            train_data=train_loader,
            test_data=test_loader,
        )

        # Gather training losses from all processes
        all_train_losses = comm.gather(train_loss, root=0)

        if rank == 0:
            avg_train_loss = sum(all_train_losses) / len(all_train_losses)
            print(f"Epoch {e} - Average Training loss: {avg_train_loss}")

    # Synchronize all processes before moving to the next epoch
    comm.Barrier()

# Evaluate and print test accuracy for each process
for i, worker_model in enumerate(worker_models):
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
    test_acc = accuracy(worker_model, head_model, test_loader)
    print(f"Process {rank} - Worker {i} Test Accuracy: {test_acc}")
