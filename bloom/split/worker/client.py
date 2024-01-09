from typing import List
from torch import nn
from torch.nn import Module
import torch.optim as optim
import torch
from bloom.models import Cifar10CNNWorkerModel, TelecomConv2DmWorkerModel
import ray

# Dictionary mapping optimizer names to their classes
OPTIMIZERS = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
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
        self.model = TelecomConv2DmWorkerModel()
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
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.train(True)
        for epoch in range(epochs):
            #     loss = 0.0
            #     for inputs, labels in self.train_data:
            #         inputs = inputs
            #         self.optimizer.zero_grad()
            #         client_output = self.model(inputs)
            #         grad_from_server, loss = ray.get(
            #             server_actor.process_and_update.remote(client_output, labels)
            #         )
            #         client_output.backward(grad_from_server)
            #         self.optimizer.step()
            #     self.losses.append(loss)
            #     if self.wandb:
            #         wandb.log({"loss": loss})

            running_loss = 0.0
            for batch, (seq, y_label) in enumerate(self.train_data):
                seq = seq
                # seq, y_label = seq.to(device), y_label.to(device)
                # print("seq shape: ", seq.shape)
                # print("y_label shape: ", y_label.shape)
                # y_label = y_label.view(-1, 1) # reshape the label

                self.optimizer.zero_grad()
                client_output = self.model(seq)
                grad_from_server, loss = ray.get(
                    server_actor.process_and_update.remote(client_output, y_label)
                )
                client_output.backward(grad_from_server)
                self.optimizer.step()

                _loss = loss
                running_loss += _loss
                self.losses.append(_loss)
                if batch % 1000 == 0:
                    print(f"Epoch {epoch+1} completed, loss: {running_loss}")

    def test(self, server_actor: ray.actor.ActorHandle) -> (float, float):
        """Tests the model using the given server.

        Args:
            server_actor (ServerActor): The server actor.

        Returns:
            avg_loss (float): The average loss.
            accuracy (float): The accuracy.

        """
        # self.model.eval()
        # total = 0
        # correct = 0
        # total_loss = 0.0
        # for inputs, labels in self.test_data:
        #     inputs = inputs
        #     client_output = self.model(inputs)
        #     loss, correct_pred, total_pred = ray.get(
        #         server_actor.validate.remote(client_output, labels)
        #     )
        #     total += total_pred
        #     correct += correct_pred
        #     total_loss += loss
        # avg_loss = total_loss / len(self.test_data)
        # accuracy = 100 * correct / total
        # return avg_loss, accuracy
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        # total = 0
        # correct = 0
        total_loss = 0.0
        preds = []
        running_mae, running_mse = 0.0, 0.0
        for x, y in self.test_data:
            x, y = x.to(device), y.to(device)
            # y = y.reshape(x.shape[0], 1).double()

            client_output = self.model(x)
            pred, error, squared_error, _loss = ray.get(
                server_actor.validate.remote(client_output, y)
            )
            preds.append(pred)
            # total += total_pred
            # correct += correct_pred
            total_loss += _loss
            running_mae += error
            running_mse += squared_error
        mae = running_mae / len(self.test_data)
        mse = running_mse / len(self.test_data)
        avg_loss = total_loss / len(self.test_data)
        # accuracy = 100 * correct / total
        accuracy = 0
        print(f"MAE value: {mae:.5f}, MSE value: {mse:.5f}")
        return avg_loss, accuracy

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
