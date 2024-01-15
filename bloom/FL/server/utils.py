import flwr as fl
import os
from typing import List
import numpy as np
import wandb
import torch
from collections import OrderedDict

from bloom import models, ROOT_DIR

IS_WANDB_TRACK = False  # <-needs to be exported to yaml


# function to get the strategy based on the name
def define_strategy(
    strat: str, wandb_track: bool, params: List[np.ndarray] = None
) -> fl.server.strategy:
    """
        Returns the strategy function based on the name

        Set up the strategy funciton based on the name and parameters
        to be used for starting the flower server.
        Available strategies: FedAvg, FedAdam, FedYogi, FedAdagrad, FedAvgM

    Args:
        strat: name of the strategy algorithm (string)
        params: parameters of the model (list of numpy arrays)

    Returns:
        strategy: the strategy function
    """

    if strat == "FedAdam":
        if params is None:
            raise ValueError("Initial model parameters missing for FedAdam")

        strategy = fl.server.strategy.FedAdam(
            fraction_fit=0.5,
            fraction_evaluate=0.5,
            min_fit_clients=3,
            min_evaluate_clients=3,
            min_available_clients=3,
            initial_parameters=fl.common.ndarrays_to_parameters(params),
            evaluate_metrics_aggregation_fn=weighted_average,
            eta=0.01,
            beta_1=0.9,
            eta_l=0.1,
        )
    elif strat == "FedAvg":
        # Federated Averaging strategy
        strategy = SaveFedAvg(evaluate_metrics_aggregation_fn=weighted_average)
    elif strat == "FedAvgM":
        # Configurable FedAvg with Momentum strategy implementation
        if params is None:
            raise ValueError("Initial model parameters missing for FedAvgM")
        strategy = fl.server.strategy.FedAvgM(
            evaluate_metrics_aggregation_fn=weighted_average,
            min_available_clients=2,
            initial_parameters=fl.common.ndarrays_to_parameters(params),
            server_learning_rate=0.1,
            server_momentum=0.9,
        )
    elif strat == "FedYogi":
        # Adaptive Federated Optimization using Yogi
        if params is None:
            raise ValueError("Initial model parameters missing for FedYogi")
        strategy = fl.server.strategy.FedYogi(
            evaluate_metrics_aggregation_fn=weighted_average,
            min_available_clients=2,
            initial_parameters=fl.common.ndarrays_to_parameters(params),
            eta=0.1,
            beta_1=0.9,
        )
    elif strat == "FedAdagrad":
        # FedAdagrad strategy - Adaptive Federated Optimization using Adagrad.
        if params is None:
            raise ValueError("Initial model parameters missing for FedAdagrad")
        strategy = fl.server.strategy.FedAdagrad(
            evaluate_metrics_aggregation_fn=weighted_average,
            min_available_clients=2,
            initial_parameters=fl.common.ndarrays_to_parameters(params),
            eta=0.1,
            eta_l=0.01,
        )

    return strategy


def get_parameters(net) -> List[np.ndarray]:
    """
    Returns the parameters of the model

    Args:
        net: the model

    Returns:
        the parameters of the model
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def weighted_average(metrics: dict) -> dict:
    """
    Returns the weighted average of the metrics

    Args:
        metrics: the metrics reported by the clients (e.g. accuracy, loss and etc.)

    Returns:
        A dictionary with the weighted average of the metrics
    """
    acc = [num_examples * m["accuracy"] for num_examples, m in metrics]
    f1_score = [num_examples * m["f1"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    if IS_WANDB_TRACK:
        # wandb logging
        wandb.log(
            {"acc": sum(acc) / sum(examples), "f1": sum(f1_score) / sum(examples)}
        )
    return {"accuracy": sum(acc) / sum(examples)}


class SaveFedAvg(fl.server.strategy.FedAvg):
    """Override strategies to save the model for final evaluation"""

    def aggregate_fit(self, server_round: int, results, failures):
        net = models.FedAvgCNN()

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(net.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

            # Save the model
            exp_folder = os.path.join(ROOT_DIR, "FL", "saved_fl_models")
            print(exp_folder)
            if not os.path.exists(exp_folder):
                os.makedirs(exp_folder)
            torch.save(net.state_dict(), f"{exp_folder}/model_round_{server_round}.pth")

        return aggregated_parameters, aggregated_metrics
