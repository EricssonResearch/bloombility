import wandb  # for tracking experiments
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from context import models
from context import load_data


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

# ----------------------------------------- classification methods ------------------------------------------------------


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


def main(config):
    """
    reads config, downloads / locally loads chosen dataset, preprocesses it,
    defines the chosen model, optimizer and loss, and starts training
    """
    dataset = config.get_chosen_datasets()
    opt = config.get_chosen_optimizers()
    loss_fun = config.get_chosen_loss("classification")
    wandb_track = config.get_wand_active()
    wandb_key = config.get_wandb_key()
    hyper_params = config.get_hyperparams()

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
                "dataset": dataset,
                "optimizer": opt,
                "epochs": hyper_params["num_epochs"],
                "loss": loss_fun,
            },
        )

    # set up transform to normalize data
    if dataset == "CIFAR10":
        # has three channels b/c it is RGB -> normalize on three layers
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif dataset == "FEMNIST":
        # has one channel b/c it is grayscale -> normalize on one layer
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    else:
        print("Unrecognized dataset")
        quit()

    if dataset == "CIFAR10":
        # download CIFAR10 training dataset and apply transform
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )

        # download CIFAR10 testing dataset and apply transform
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )

    elif dataset == "FEMNIST":
        # download FEMNIST training dataset and apply transform
        trainset = load_data.download_femnist.FEMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        # download FEMNIST testing dataset and apply transform
        testset = load_data.download_femnist.FEMNIST(
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
    if dataset == "CIFAR10":
        model = models.Networks.CNNCifar(num_CIFAR_classes).to(device)
    elif dataset == "FEMNIST":
        model = models.Networks.CNNFemnist(num_FEMNIST_classes).to(device)
    else:
        print("did not recognized chosen NN model. Check your constants.")
        quit()

    # Setting the loss function
    if loss_fun == "CrossEntropyLoss":
        cost = nn.CrossEntropyLoss()
    elif loss_fun == "NLLLoss":
        cost = nn.NLLLoss()
    else:
        print("Unrecognized loss funct")
        quit()

    # Setting the optimizer with the model parameters and learning rate
    if opt == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif opt == "Adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif opt == "Adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif opt == "RMSProp":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=hyper_params["learning_rate"]
        )
    elif opt == "SGD":
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
