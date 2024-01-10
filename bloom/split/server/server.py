from typing import List, Dict, Tuple
from torch import nn
from torch.optim import Optimizer
import torch.optim as optim
import torch

# Import the model you want to use based on models/Networks.py
from bloom.models import Cifar10CNNHeadModel, CNNFemnistHeadModel
import ray


# Dictionary mapping optimizer names to their classes
OPTIMIZERS: Dict[str, Optimizer] = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}


@ray.remote  # Specify the number of GPUs the actor should use
class ServerActor:
    """
    The ServerActor class represents the server in a split learning setup.
    It contains the tail of the model and is responsible for aggregating the gradients from the workers and updating the model.

    Attributes:
        model (Module): The model used for training.
        criterion (CrossEntropyLoss): The loss function used for training.
        optimizer (Optimizer): The optimizer used for training.
    """

    def __init__(self, config: Dict = {}):
        """
        Initializes the ServerActor with a given configuration.

        Args:
            config (Dict): The configuration for the ServerActor.

        Returns:
            Object: The ServerActor object.

        """
        self.model = Cifar10CNNHeadModel()
        self.criterion = nn.CrossEntropyLoss()
        # Create the optimizer using the configuration parameters
        OptimizerClass = OPTIMIZERS[config.optimizer]
        self.optimizer = OptimizerClass(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

    def process_and_update(self, client_output: torch.Tensor, labels: torch.Tensor):
        """
        Processes the output from the worker client, runs it through the rest of the model and updates the model parameters.

        Args:
            client_output (torch.Tensor): The output from the worker client.
            labels (torch.Tensor): The labels for the data.

        Returns:
            Tuple[torch.Tensor, float]: The gradient of the client output and the loss value.
        """
        self.optimizer.zero_grad()
        labels = labels
        output = self.model(client_output)
        loss = self.criterion(output, labels)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        return client_output.grad, loss.item()

    def validate(self, client_output: torch.Tensor, labels: torch.Tensor):
        """
        Validates the model with the given client output and labels.

        Args:
            client_output (torch.Tensor): The output from the client.
            labels (torch.Tensor): The labels for the data.

        Returns:
            Tuple[float, int, int]: The loss value, the number of correct predictions, and the total number of predictions.
        """
        with torch.no_grad():
            labels = labels
            output = self.model(client_output)
            loss = self.criterion(output, labels)
            _, predicted = torch.max(output.data, 1)
            total = labels.size(0)
            correct = (predicted == labels).sum().item()
        return loss.item(), correct, total, predicted
