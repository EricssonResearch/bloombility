import wandb  # for tracking experiments
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from bloom import models
from bloom import load_data


# based on tutorial here: https://blog.paperspace.com/writing-cnns-from-scratch-in-pytorch/


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

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------- dataset ------------------------------------------------------
def get_preprocessed_FEMNIST():
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


def get_preprocessed_CIFAR10():
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


def transform_to_loader(data, hyper_params):
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


def get_classification_loaders(_dataset, hyper_params):
    """based on the chosen dataset, retrieve the data, pre-process it
    and convert it into a DataLoader

    Params:
        _dataset: chosen dataset

    Returns:
        trainloader: training data DataLoader object
        testloader: testing data DataLoader object
    """
    # set up data
    if _dataset == "CIFAR10":
        trainset, testset = get_preprocessed_CIFAR10()
    elif _dataset == "FEMNIST":
        trainset, testset = get_preprocessed_FEMNIST()
    else:
        print("Unrecognized dataset")
        quit()

    trainloader = transform_to_loader(trainset, hyper_params)
    testloader = transform_to_loader(testset, hyper_params)

    return trainloader, testloader


# ----------------------------------------- config -------------------------------------------------------


def get_classification_optimizer(_opt, model, hyper_params):
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


def get_classification_model(_dataset):
    """based on the chosen dataset, return correct model
    Params:
        _dataset: chosen dataset

    Returns:
        model: model as defined in Networks file
    """
    # setting up model
    if _dataset == "CIFAR10":
        model = models.Networks.CNNCifar(num_CIFAR_classes).to(device)
    elif _dataset == "FEMNIST":
        model = models.Networks.CNNFemnist(num_FEMNIST_classes).to(device)
    else:
        print("did not recognized chosen NN model. Check your constants.")
        quit()
    return model


def get_classification_loss(_loss):
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


# ----------------------------------------- training & testing --------------------------------------------
def training(trainloader, testloader, model, num_epochs, optimizer, cost, wandb_track):
    """
    trains the model on training dataset

    for each epoch, do the following:
        present each image from the training dataset to the model and save its output label.
        With the loss funct, calculate how far off the expected result is from the actual result.
        Propagate the difference of model weights backwards through the model to improve classification.
        Repeat this procedure for each image in the training dataset.

    Afterwards, call classification_accuracy to estimate model performance on test set.

    Args:
        trainloader: the preprocessed training set in a lightweight format
        testloader: the preprocessed testing set in a lightweight format
        model: the NN model to be trained
        optimizer: the optimizer to update the model with
        cost: the loss function to calculate the difference between expected and actual result

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

            if (i + 1) % 400 == 0:
                print(
                    "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                        epoch + 1, num_epochs, i + 1, total_step, loss.item()
                    )
                )

        # set to evaluation mode
        model.eval()
        acc_per_epoch = classification_accuracy(testloader, model, epoch)

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


def classification_accuracy(testloader, model, epoch):
    """
    evaluates accuracy of network on train dataset

    compares expected with actual output of the model
    when presented with images from previously unseen testing set.
    This ensures that the model does not just "know the training data results by heart",
    but has actually found and learned patterns in the training data

    Args:
        testloader: the preprocessed testing set in a lightweight format
        model: the pretrained(!) NN model to be evaluated

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

        print(
            "Accuracy of the network after epoch {} on the {} test images: {} %".format(
                epoch + 1, 50000, acc
            )
        )

        return acc


def main(config):
    """
    reads config, downloads / locally loads chosen dataset, preprocesses it,
    defines the chosen model, optimizer and loss, and starts training
    """
    _dataset = config.get_chosen_datasets("classification")
    _opt = config.get_chosen_optimizers("classification")
    _loss = config.get_chosen_loss("classification")
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

    # set up data
    trainloader, testloader = get_classification_loaders(_dataset, hyper_params)

    # config
    model = get_classification_model(_dataset)
    cost = get_classification_loss(_loss)
    optimizer = get_classification_optimizer(_opt, model, hyper_params)

    # start training process
    training(
        trainloader,
        testloader,
        model,
        hyper_params["num_epochs"],
        optimizer,
        cost,
        wandb_track,
    )


# call main function when running the script
if __name__ == "__main__":
    main()
