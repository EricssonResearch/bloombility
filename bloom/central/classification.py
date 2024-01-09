import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from bloom import models
from bloom import load_data
from bloom.load_data.data_distributor import DATA_DISTRIBUTOR


# based on tutorial here: https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/
class data_split_config:
    """for testing purposes:
    Get data via data_distributor module
    """

    dirichlet_alpha = 1
    niid_factor = 2


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


# ----------------------------------------- dataset ------------------------------------------------------
def get_preprocessed_FEMNIST() -> (
    tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
):
    """ "downloads and transforms FEMNIST

    Returns:
        trainset: Training dataset
        testset: Testing dataset
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    # download FEMNIST training dataset and apply transform
    trainset = load_data.download_femnist.FEMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    # download FEMNIST testing dataset and apply transform
    testset = load_data.download_femnist.FEMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    return trainset, testset


def get_preprocessed_CIFAR10() -> (
    tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
):
    """ "downloads and transforms CIFAR10

    Returns:
        trainset: Training dataset
        testset: Testing dataset
    """
    # has three channels b/c it is RGB -> normalize on three layers
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # download CIFAR10 training dataset and apply transform
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    # download CIFAR10 testing dataset and apply transform
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    return trainset, testset


def transform_to_loader(
    data: torch.utils.data.Dataset, hyper_params: dict
) -> torch.utils.data.DataLoader:
    """get a dataset, convert it into a torch DataLoader

    Params:
        data: dataset to convert
        hyperparams: configuration dictionary including batch size and number of workers
    Returns:
        DataLoader object
    """
    return torch.utils.data.DataLoader(
        data,
        batch_size=hyper_params["batch_size"],
        shuffle=True,
        num_workers=hyper_params["num_workers"],
    )


def get_classification_loaders(
    _dataset: str, hyper_params: dict
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """based on the chosen dataset, retrieve the data, pre-process it
    and convert it into a DataLoader

    Params:
        _dataset: chosen dataset
        hyper_params: dict of configurations, which has to include num_workers and batch_size

    Returns:
        trainloader: training data DataLoader object
        testloader: testing data DataLoader object
    """
    # set up data
    if _dataset == "CIFAR10":
        trainset, testset = get_preprocessed_CIFAR10()
    elif _dataset == "FEMNIST":
        trainset, testset = get_preprocessed_FEMNIST()
    elif _dataset == "IID":
        # for testing purposes: non-iid data splits via data_distrubutor module
        cl = data_split_config
        data_distributor = DATA_DISTRIBUTOR(4, cl, "iid")
        testloader = data_distributor.get_testloader()
        trainloaders = data_distributor.get_trainloaders()
        print("Amount of loaders:", len(trainloaders))
        for i, loader in enumerate(trainloaders):
            print(
                f"Length of trainloader {i}: {len(loader)} with batch size 32 equals approx. {len(loader)*32} total images"
            )
        # because it's centralized, only needs one trainloader
        trainloader = trainloaders[0]
        return trainloader, testloader
    else:
        print("Unrecognized dataset")
        quit()

    trainloader = transform_to_loader(trainset, hyper_params)
    testloader = transform_to_loader(testset, hyper_params)

    return trainloader, testloader


# ----------------------------------------- config -------------------------------------------------------


def get_classification_optimizer(
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
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif _opt == "Adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif _opt == "Adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif _opt == "RMSProp":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif _opt == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    else:
        print("Unrecognized optimizer!")
        quit()
    return optimizer


def get_classification_model(_dataset: str, device: str) -> nn.Module:
    """based on the chosen dataset, return correct model

    Params:
        _dataset: chosen dataset
        device: where calculations are performed (cuda, which means gpu / cpu)

    Returns:
        model: model as defined in Networks file
    """
    # setting up model
    if _dataset == "CIFAR10" or _dataset == "IID":
        # IID here means it's the CIFAR10 dataset accessed through data_distributor
        model = models.Networks.CNNCifar(num_CIFAR_classes).to(device)
    elif _dataset == "FEMNIST":
        model = models.Networks.CNNFemnist(num_FEMNIST_classes).to(device)
    else:
        print("did not recognize chosen NN model. Check your constants.")
        quit()
    return model


def get_classification_loss(_loss: str) -> nn:
    """based on the chosen loss, return correct loss function object
    Params:
        _loss: chosen loss

    Returns:
        cost: loss function object
    """
    # Setting the loss function
    if _loss == "CrossEntropyLoss":
        cost = nn.CrossEntropyLoss()
    elif _loss == "NLLLoss":
        cost = nn.NLLLoss()
    else:
        print("Unrecognized loss funct")
        quit()
    return cost


# --------------------------------- accuracy ----------------------------------------------------


def classification_accuracy(
    testloader: torch.utils.data.DataLoader, model: nn.Module, device: str
) -> float:
    """
    evaluates accuracy of network on test dataset

    compares expected with actual output of the model
    when presented with images from previously unseen testing set.
    This ensures that the model does not just "know the training data results by heart",
    but has actually found and learned patterns in the training data

    Params:
        testloader: the preprocessed testing set in a lightweight format
        model: the pretrained(!) NN model to be evaluated
        device: where calculations are performed (cuda, which means gpu / cpu)

    Returns:
        acc: accuracy of classification

    """
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total

        return acc
