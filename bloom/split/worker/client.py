from typing import List
from torch import nn
from torch.nn import Module
import torch.optim as optim
import numpy as np
import torch
from bloom.models import Cifar10CNNWorkerModel, CNNFemnistWorkerModel
import ray
import wandb
from sklearn.metrics import f1_score
from sklearn.preprocessing import label_binarize


# Dictionary mapping optimizer names to their classes
OPTIMIZERS = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
}

# Dictionary mapping dataset names to their model classes
MODELS = {
    "CIFAR10": Cifar10CNNWorkerModel,
    "FEMNIST": CNNFemnistWorkerModel,
}


@ray.remote
class WorkerActor:
    """A class representing a worker in a distributed learning setup. The worker holds the model head and communicates with the server to update the model.

    Attributes:
        model (Module): The model used for learning.
        train_data (Dataset): The training data.
        test_data (Dataset): The testing data.
        optimizer (Optimizer): The optimizer used for learning.
        losses (list): A list to store the loss values.
        wandb (bool): A flag indicating whether to use Weights & Biases for tracking.

    """

    def __init__(
        self,
        train_data: torch.utils.data,
        test_data: torch.utils.data,
        input_layer_size: int,
        wandb: bool = False,
        config: dict = {},
    ) -> None:
        """Initializes the WorkerActor with the given data, model input size and optimizer.

        Args:
            train_data (Dataset): The training data.
            test_data (Dataset): The testing data.
            input_layer_size (int): The size of the input layer of the model.
            wandb (bool, optional): A flag indicating whether to use Weights & Biases for tracking. Defaults to False.
            config (dict, optional): A configuration dictionary. Defaults to {}.
        Returns:
            Object: The WorkerActor object.

        """
        ModelClass = MODELS[config.dataset]
        self.model = ModelClass(input_layer_size)
        self.train_data = train_data
        self.test_data = test_data
        # Create the optimizer using the configuration parameters
        OptimizerClass = OPTIMIZERS[config.optimizer]
        self.optimizer = OptimizerClass(
            self.model.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
        self.losses = []
        self.wandb = wandb

    def getattr(self, attr: str) -> object:
        """Returns the value of the given attribute.

        Args:
            attr (str): The name of the attribute.

        Returns:
            The value of the attribute.

        """
        return getattr(self, attr)

    def train(self, server_actor: ray.actor.ActorHandle, epochs: int) -> None:
        """Trains the model using the given server and number of epochs.

        Args:
            server_actor (ServerActor): The server actor.
            epochs (int): The number of epochs.

        Returns:
            None

        """
        for epoch in range(epochs):
            loss = 0.0
            for inputs, labels in self.train_data:
                inputs = inputs
                self.optimizer.zero_grad()
                client_output = self.model(inputs)
                grad_from_server, loss = ray.get(
                    server_actor.process_and_update.remote(client_output, labels)
                )
                client_output.backward(grad_from_server)
                self.optimizer.step()
            self.losses.append(loss)
            if self.wandb:
                wandb.log({"loss": loss})
            print(f"Epoch {epoch+1} completed, loss: {loss}")

    def test(self, server_actor: ray.actor.ActorHandle) -> (float, float):
        """Tests the model using the given server.

        Args:
            server_actor (ServerActor): The server actor.

        Returns:
            avg_loss (float): The average loss.
            accuracy (float): The accuracy.

        """
        total = 0
        correct = 0
        total_loss = 0.0
        y_true = []
        y_pred = []

        all_labels = []
        all_scores = []

        for inputs, labels in self.test_data:
            inputs = inputs
            client_output = self.model(inputs)
            loss, correct_pred, total_pred, predicted, output = ray.get(
                server_actor.validate.remote(client_output, labels)
            )
            # Convert the output scores to probabilities
            scores = torch.nn.functional.softmax(output, dim=1).cpu().numpy()
            all_scores.append(scores)

            # Convert the target labels to a binary format
            _labels = label_binarize(labels.cpu().numpy(), classes=np.arange(10))
            all_labels.append(_labels)

            total += total_pred
            correct += correct_pred
            total_loss += loss
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
        avg_loss = total_loss / len(self.test_data)
        accuracy = 100 * correct / total
        f1 = f1_score(y_true, y_pred, average="weighted")
        # Concatenate all the scores and labels
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        return avg_loss, accuracy, f1, all_labels, all_scores

    def get_model(self) -> Module:
        """Returns the model.

        Returns:
            Module (nn.Module): The model.

        """
        return self.model

    def get_weights(self) -> torch.Tensor:
        """Returns the weights of the model.

        Returns:
            torch.Tensor: The weights of the model.

        """
        return self.model.state_dict()

    def set_weights(self, weights: torch.Tensor) -> nn.Module:
        """Sets the weights of the model.

        Args:
            weights (torch.Tensor): The new weights.

        Returns:
            Module (nn.Module): The model with the new weights.
        """
        self.model.load_state_dict(weights)
        return self.model.state_dict()
