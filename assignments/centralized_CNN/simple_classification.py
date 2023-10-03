import yaml
import sys
import os
import wandb  # for tracking experiments
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from Networks import CNNCifar, CNNFemnist
from download_femnist import FEMNIST

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
default_config = "default_config.yaml"

# ----------------------------------------- methods ------------------------------------------------------


def training(trainloader, testloader, model, num_epochs, optimizer, cost, wandb_track):
    """
    trains the model on training dataset

    for each epoch, do the following:
        present each image from the training dataset to the model and save its output label.
        With the loss funct, calculate how far off the expected result is from the actual result.
        Propagate the difference of model weights backwards through the model to improve classification.
        Repeat this procedure for each image in the training dataset.

    Afterwards, call eval_results to estimate model performance on test set.

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
        acc_per_epoch = eval_results(testloader, model, epoch)

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


def eval_results(testloader, model, epoch):
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
        config["wandb_tracking"]["activated"],
        config["hyper-params"],
    )


def main():
    """
    reads config, downloads / locally loads chosen dataset, preprocesses it,
    defines the chosen model, optimizer and loss, and starts training
    """
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = os.path.join(os.path.dirname(__file__), default_config)
    # config_file = os.path.join(os.getcwd(), 'assignments', 'centralized_CNN', 'config.yaml')
    config = read_config_file(config_file)
    which_dataset, which_opt, which_loss, wandb_track, hyper_params = parse_config(
        config
    )

    if wandb_track:
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="bloomnet_visualization",
            # track hyperparameters and run metadata
            config={
                "learning_rate": hyper_params["learning_rate"],
                "dataset": which_dataset,
                "optimizer": which_opt,
                "epochs": hyper_params["num_epochs"],
                "loss": which_loss,
            },
        )

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
        quit()

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
        trainset,
        batch_size=hyper_params["batch_size"],
        shuffle=True,
        num_workers=hyper_params["num_workers"],
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=hyper_params["batch_size"],
        shuffle=False,
        num_workers=hyper_params["num_workers"],
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
    if which_loss == "CrossEntropyLoss":
        cost = nn.CrossEntropyLoss()
    elif which_loss == "NLLLoss":
        cost = nn.NLLLoss()
    else:
        print("Unrecognized loss funct")
        quit()

    # Setting the optimizer with the model parameters and learning rate
    if which_opt == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif which_opt == "Adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif which_opt == "Adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif which_opt == "RMSProp":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif which_opt == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    else:
        print("Unrecognized optimizer!")
        quit()

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
