import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from Networks import ConvNeuralNet


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


# Define relevant variables for the ML task
batch_size = 4
num_CIFAR_classes = len(CIFAR10_classes)
learning_rate = 0.01
num_epochs = 5
num_workers = 2

# Device will determine whether to run the training on GPU or CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    eval_results(trainloader, testloader, model)


def eval_results(trainloader, testloader, model):
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


def main():
    # set up transform to normalize data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # download CIFAR10 training dataset and apply transform
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    # download CIFAR10 testing dataset and apply transform
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # setting up model
    model = ConvNeuralNet(num_CIFAR_classes).to(device)

    # Setting the loss function
    cost = nn.CrossEntropyLoss()

    # Setting the optimizer with the model parameters and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training(trainloader, testloader, model, optimizer, cost)


if __name__ == "__main__":
    main()
    print(num_CIFAR_classes)
