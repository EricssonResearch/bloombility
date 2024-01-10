# Run: python bloom/split/main.py --num-workers 2
#
#   This is the main entry point for the split learning module.
#   It is responsible for instantiating the server and worker actors,
#   and starting the training and testing processes.
#
#   Written by Moji @ 2023


from copy import deepcopy
from typing import List
import torch
from torch.nn import CrossEntropyLoss
import numpy as np
import matplotlib.pyplot as plt
from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
from bloom import ROOT_DIR
import argparse
import os
from datetime import datetime
import wandb

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import precision_recall_curve, average_precision_score
from itertools import cycle

config_path = os.path.join(ROOT_DIR, "config", "split")

os.environ["RAY_DEDUP_LOGS"] = "0"  # Disable deduplication of RAY logs
import ray

from worker import WorkerActor
from server import ServerActor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = None


def init_wandb(num_workers: int, conf: dict = {}) -> None:
    """
    Initialize wandb for logging.

    Args:
        num_workers (int): Number of workers.
        conf (dict): Configuration dictionary. Requires the WANDB_API_KEY environment variable to be set.

    Returns:
        None
    """
    wandb_key = conf["api_key"]
    wandb.login(anonymous="never", key=wandb_key)
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        entity=conf["entity"],
        project=conf["project"],
        group=conf["group"],
        # track hyperparameters and run metadata
        config={
            "method": "split",
            "n_epochs": EPOCHS,
            "n_workers": num_workers,
        },
    )


def plot_workers_losses(
    workers: list, wandb_track: bool = False, showPlot: bool = False
) -> None:
    """
    Plot the losses of each worker.

    Args:
        workers (list): List of worker actors.
        wandb_track (bool, optional): Whether or not to track and visualize experiment performance with Weights and Biases. Defaults to False.
        showPlot (bool, optional): Whether or not to show the plot. Defaults to False.

    Returns:
        None
    """

    losses_future = [worker.getattr.remote("losses") for worker in workers]
    losses = ray.get(losses_future)
    plt.figure(figsize=(6.4 * 2, 4.8 * 2))
    for worker_losses in losses:
        plt.plot(worker_losses)
    plt.xlabel("Epoch")
    # set x-axis label to be the epoch number (only integers)
    plt.xticks(np.arange(0, len(worker_losses), 1.0))
    plt.ylabel("Loss")
    plt.title("Workers Losses")
    # set plot legend to be the worker number
    plt.legend([f"Worker {i+1}" for i in range(len(workers))])
    # Get current time
    now = datetime.now()
    # Format as string (YYYYMMDD_HHMMSS format)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    # Create plots directory if it does not exists
    if not os.path.exists(f"{ROOT_DIR}/split/plots/"):
        os.makedirs(f"{ROOT_DIR}/split/plots/")
    plt.savefig(f"{ROOT_DIR}/split/plots/workers_losses_{timestamp_str}.png")
    if wandb_track:
        wandb.log({"workers_losses": plt})
    if showPlot:
        plt.show()


def plot_precision_recall(
    y_test,
    y_score,
    wandb_track: bool = False,
    showPlot: bool = False,
    dataset: str = "CIFAR10",
):
    precision = dict()
    recall = dict()
    average_precision = dict()
    n_classes = 10

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
    plt.figure(figsize=(6.4 * 2, 4.8 * 2))
    plt.plot(
        recall["micro"],
        precision["micro"],
        color="gold",
        lw=2,
        label="micro-average (area = {0:0.2f})" "".format(average_precision["micro"]),
    )

    # Set colors for each class (10 classes)
    colors = cycle(
        [
            "aqua",
            "darkorange",
            "cornflowerblue",
            "red",
            "green",
            "blue",
            "yellow",
            "purple",
            "pink",
            "black",
        ]
    )
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            label="Class {0} (area = {1:0.2f})" "".format(i, average_precision[i]),
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("10-class Precision-Recall curve for CIFAR-10")
    plt.legend(loc="lower right")
    # Get current time
    now = datetime.now()
    # Format as string (YYYYMMDD_HHMMSS format)
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{ROOT_DIR}/split/plots/precision_recall_curve_{timestamp_str}.png")
    if wandb_track:
        wandb.log({"precision_recall_curve": plt})
    if showPlot:
        plt.show()


@hydra.main(config_path=config_path, config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the split learning module.
    Performs the following steps:
    0. Read configuration file, instantiate wandb, and initialize Ray.
    1. Load data using the data distributor class.
    2. Instantiate the server and models.
    3. Spawn server and worker actors.
    4. Start sequential training and testing processes.
    5. Aggregate test results.
    6. Plot the losses of each worker.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        None
    """
    # Use argparse to get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2, help="number of workers")

    args = parser.parse_args()

    # PARAMS from config
    MAX_CLIENTS = cfg.main.max_clients
    EPOCHS = cfg.main.n_epochs

    if cfg.main.n_workers != args.num_workers:
        num_workers = cfg.main.n_workers
    else:
        num_workers = args.num_workers

    showPlot = cfg.main.show_plot
    parallel_training = cfg.main.parallel_training
    # set Dataset from main config file to server and worker config files
    cfg.server.dataset = cfg.worker.dataset = cfg.main.dataset

    if num_workers > MAX_CLIENTS:
        raise ValueError(
            "Number of clients must be less than or equal to ", MAX_CLIENTS
        )

    # WANDB experiments

    # Enable wandb tracking (True/False)
    wandb_track = cfg.main.wandb.track

    wandb_config = {
        "api_key": os.environ["WANDB_API_KEY"],
        "entity": cfg.main.wandb.entity,
        "project": cfg.main.wandb.project,
        "group": cfg.main.wandb.group,
    }
    if wandb_track:
        init_wandb(num_workers, wandb_config)

    # Load data using the data distributor class
    data_distributor = None
    if data_distributor is None:
        data_distributor = DATA_DISTRIBUTOR(num_workers, dataset=cfg.main.dataset)
        trainloaders = data_distributor.get_trainloaders()
        test_data = data_distributor.get_testloader()

    # Shut down Ray if it has already been initialized
    if ray.is_initialized():
        ray.shutdown()
    # Instantiate the server and models
    ray_ctx = ray.init(namespace="split_learning", num_gpus=1)

    print("============================== INFO ==============================")
    main_node_address = ray_ctx.address_info["redis_address"]
    print(f"Ray initialized with address: {main_node_address}")
    cluster_resources = ray.cluster_resources()
    print(f"Ray initialized with resources: {cluster_resources}")
    print("============================== END ==============================")

    # Spawn server and worker actors
    server = ServerActor.remote(config=cfg.server)

    input_layer_size = cfg.worker.input_layer_size
    workers = [
        WorkerActor.options(name=f"worker_{i}", namespace="split_learning").remote(
            trainloaders[i], test_data, input_layer_size, config=cfg.worker
        )
        for i in range(num_workers)
    ]

    if parallel_training:
        # ==== Parallel training and testing processes (No weight sharing) ====#
        # Start training on each worker (in parallel)
        train_futures = [worker.train.remote(server, EPOCHS) for worker in workers]
        ray.get(train_futures)  # Wait for training to complete

        # Start testing on each worker
        test_futures = [worker.test.remote(server) for worker in workers]
        test_results = ray.get(test_futures)

        # Start training on each worker and wait for it to complete before moving to the next
        for worker in workers:
            train_future = worker.train.remote(server, EPOCHS)
            ray.get(train_future)

    else:
        # ==== Sequential training and testing processes (With weight sharing) ====#
        for i in range(len(workers) - 1):
            # Train the current worker
            train_future = workers[i].train.remote(server, EPOCHS)
            ray.get(train_future)
            # Get the weights from the trained worker
            weights = ray.get(workers[i].get_weights.remote())

            # Set the weights of the next worker to the weights of the current worker
            workers[i + 1].set_weights.remote(weights)

        # Train the last worker
        train_future = workers[-1].train.remote(server, EPOCHS)
        ray.get(train_future)

        # Start testing on each worker and wait for it to complete before moving to the next
        test_results = []
        y_test = []
        y_score = []
        for worker in workers:
            test_future = worker.test.remote(server)
            test_results.append(ray.get(test_future))
            y_test.append(test_results[-1][3])
            y_score.append(test_results[-1][4])
        # Convert lists to numpy arrays
        y_test = np.concatenate(y_test)
        y_score = np.concatenate(y_score)

        # == Precision-Recall Curve == #
        if cfg.main.dataset == "CIFAR10":
            plot_precision_recall(y_test, y_score, wandb_track, showPlot)

    # Aggregate test results
    avg_loss = sum([result[0] for result in test_results]) / len(test_results)
    avg_accuracy = sum([result[1] for result in test_results]) / len(test_results)
    avg_f1 = sum([result[2] for result in test_results]) / len(test_results)
    print("Accuracies: ", [result[1] for result in test_results])
    print("F1 Scores: ", [result[2] for result in test_results])
    print(f"Average Test Loss: {avg_loss}\nAverage Accuracy: {avg_accuracy}%")
    print(f"Average F1 Score: {avg_f1}")

    plot_workers_losses(workers, wandb_track, showPlot)

    if wandb_track:
        wandb.log(
            {
                "accuracies": [result[1] for result in test_results],
                "f1_scores": [result[2] for result in test_results],
                "avg_loss": avg_loss,
                "avg_accuracy": avg_accuracy,
                "avg_f1": avg_f1,
            }
        )
        wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
