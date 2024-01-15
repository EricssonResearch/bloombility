import flwr as fl
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import wandb

from bloom import ROOT_DIR
from sklearn.metrics import precision_recall_curve, average_precision_score

IS_WANDB_TRACK = True  # <-needs to be exported to yaml


# function to get the strategy based on the name
def define_strategy(
    strat: str, wandb_track: bool, params: List[np.ndarray] = None
) -> fl.server.strategy:
    """
        Returns the strategy function based on the name

        Set up the strategy funciton based on the name and parameters
        to be used for starting the flower server.
        Available strategies: FedAvg, FedAdam, FedYogi, FedAdagrad, FedAvgM, CustomFedAvg

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
        strategy = fl.server.strategy.FedAvg(
            evaluate_metrics_aggregation_fn=weighted_average
        )
    elif strat == "CustomFedAvg":
        # Custom Federated Averaging strategy
        strategy = CustomFedAvg(evaluate_metrics_aggregation_fn=weighted_average)
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
    loss = [num_examples * m["loss"] for num_examples, m in metrics]
    precision = [num_examples * m["precision"] for num_examples, m in metrics]
    recall = [num_examples * m["recall"] for num_examples, m in metrics]
    f1_score = [num_examples * m["f1"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    if IS_WANDB_TRACK:
        # wandb logging
        wandb.log(
            {
                "acc": sum(acc) / sum(examples),
                "f1": sum(f1_score) / sum(examples),
                "loss": sum(loss) / sum(examples),
                "precision": sum(precision) / sum(examples),
                "recall": sum(recall) / sum(examples),
            }
        )
    elif not IS_WANDB_TRACK:
        print(
            f"acc: {sum(acc) / sum(examples)}, f1: {sum(f1_score) / sum(examples)}, loss: {sum(loss) / sum(examples)}, precision: {sum(precision) / sum(examples)}, recall: {sum(recall) /sum(examples)}"
        )

    # plot_precision_recall(precision, recall, examples)

    return {"accuracy": sum(acc) / sum(examples)}


def plot_precision_recall(
    y_test,
    y_score,
    wandb_track: bool = False,
    showPlot: bool = False,
):
    precision = dict()
    recall = dict()
    average_precision = dict()
    # Number of classes (10 for CIFAR-10, 62 for FEMNIST)
    n_classes = 62

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_test.ravel(), y_score.ravel()
    )
    average_precision["micro"] = average_precision_score(
        y_test, y_score, average="micro"
    )

    # Plot the micro-averaged Precision-Recall curve
    # plt.figure(figsize=(6.4 * 2, 4.8 * 2))
    plt.plot(
        recall["micro"],
        precision["micro"],
        color="gold",
        lw=2,
        label="micro-average (area = {0:0.2f})" "".format(average_precision["micro"]),
    )

    # # Dedine list of colors for plotting
    # colors = cycle(
    #     [
    #         "aqua",
    #         "darkorange",
    #         "cornflowerblue",
    #         "red",
    #         "green",
    #         "blue",
    #         "yellow",
    #         "purple",
    #         "pink",
    #         "black",
    #     ]
    # )
    # if dataset == "CIFAR10":
    #     for i, color in zip(range(n_classes), colors):
    #         plt.plot(
    #             recall[i],
    #             precision[i],
    #             color=color,
    #             lw=2,
    #             label="Class {0} (area = {1:0.2f})" "".format(i, average_precision[i]),
    #         )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Micro-averaged Precision-Recall curve - FEMNIST")
    plt.legend(loc="lower right")
    # Get current time
    now = datetime.now()
    # Format as string (YYYYMMDD_HHMMSS format)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(f"{ROOT_DIR}/split/plots/"):
        os.makedirs(f"{ROOT_DIR}/split/plots/")
    plt.savefig(
        f"{ROOT_DIR}/FL/plots/precision_recall_curve_FEMNIST_{timestamp_str}.png"
    )
    if wandb_track:
        wandb.log({"precision_recall_curve": plt})
    if showPlot:
        plt.show()


class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, results, failures):
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)

        # Collect predictions and true labels from all clients
        y_test = []
        y_score = []
        print("results length: ", len(results))
        # print shape of results list
        print("results shape: ", np.shape(results))
        for _, result in results:
            client_pred = eval(result.metrics["predictions"])  # convert string to list
            client_true = eval(result.metrics["true_labels"])  # convert string to list
            y_test.append(client_true)
            y_score.append(client_pred)

        print("y_test length: ", len(y_test))
        print("y_score length: ", len(y_score))
        # print 10 first elements of y_test and y_score
        # Convert to numpy arrays
        y_test = np.concatenate(y_test)
        y_score = np.concatenate(y_score)
        print("y_test shape: ", np.shape(y_test))
        print("y_score shape: ", np.shape(y_score))

        # Plot precision-recall curve
        plot_precision_recall(y_test, y_score, IS_WANDB_TRACK)

        return aggregated_result
