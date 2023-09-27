import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from Networks import ConvNeuralNet, CNNCifar, CNNFemnist
from download_femnist import FEMNIST

# vvvv ---------- do not change ---------------------------- vvvvv
CIFAR10_classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

num_CIFAR_classes = len(CIFAR10_classes)
num_FEMNIST_classes = 10
# ^^^^ --------------do not change ---------------------------- ^^^^

# ----------------------------------------- constants: TODO: export to yaml file ------------------------------------------

# define statics
available_datasets = {"FEMNIST", "CIFAR10"}
which_dataset = "FEMNIST"

available_optimizers = {"Adam", "Adagrad", "Adadelta", "RMSProp", "SGD"}
which_opt = "Adadelta"

available_loss_functs = {}

# Define relevant variables for the ML task
batch_size = 4
learning_rate = 0.001
num_epochs = 50
num_workers = 2


# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------------------- methods ------------------------------------------------------

"""
    trains the model on training dataset
"""


def training(trainloader, testloader, model, optimizer, cost):
    # this is defined to print how many steps are remaining when training
    total_step = len(trainloader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = cost(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 400 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

    eval_results_on_train(trainloader, testloader, model)


"""
    evaluates accuracy of network on train dataset
"""


def eval_results_on_train(trainloader, testloader, model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(
            "Accuracy of the network on the {} train images: {} %".format(
                50000, 100 * correct / total
            )
        )


""" 
    downloads / locally loads chosen dataset, preprocesses it, 
    defines the chosen model, optimizer and loss, and starts training
"""


def main():
    # set up transform to normalize data
    if which_dataset == "CIFAR10":
        # has three channels b/c it is RGB -> normalize on three layers
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif which_dataset == "FEMNIST":
        # has one channel b/c it is grayscale -> normalize on one layer
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    else:
        print("Unrecognized dataset")

    if which_dataset == "CIFAR10":
        # download CIFAR10 training dataset and apply transform
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

        # download CIFAR10 testing dataset and apply transform
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

    elif which_dataset == "FEMNIST":
        # download FEMNIST training dataset and apply transform
        trainset = FEMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        # download FEMNIST testing dataset and apply transform
        testset = FEMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    else:
        print("unrecognized option for dataset")
        quit()

    # torch applies multithreading, shuffling and batch learning
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # setting up model
    if which_dataset == "CIFAR10":
        model = CNNCifar(num_CIFAR_classes).to(device)
    elif which_dataset == "FEMNIST":
        model = CNNFemnist(num_FEMNIST_classes).to(device)
    else:
        print("did not recognized chosen NN model. Check your constants.")
        quit()

    # Setting the loss function
    cost = nn.CrossEntropyLoss()

    # Setting the optimizer with the model parameters and learning rate
    if which_opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif which_opt == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif which_opt == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif which_opt == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif which_opt == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    else:
        print("Unrecognized optimizer!")

    # start training process
    training(trainloader, testloader, model, optimizer, cost)


if __name__ == "__main__":
    main()
