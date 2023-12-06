from copy import deepcopy
import torch
from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
import argparse
import os
from bloom.models import FedAvgCNN

os.environ["RAY_DEDUP_LOGS"] = "0"  # Disable deduplication of RAY logs
import ray

from client import ParticipantActor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CLIENTS = 10
EPOCHS = 1
N_ITERATIONS = 10


def set_to_zero(model):
    for layer_weights in model.parameters():
        layer_weights.data.sub_(layer_weights.data)


def params_from_model(model):
    params = list(model.parameters())
    return [tensors.detach() for tensors in params]


def average(model, client_params, weights):
    # we obtain a new model
    new_model = deepcopy(model)

    # we set all neural parameters to zero
    set_to_zero(new_model)

    # for every participant
    for i, client_param in enumerate(client_params):
        # for every layer of each participant
        for idx, layer_weights in enumerate(new_model.parameters()):
            # calculate the contribution
            contribution = client_param[idx].data * weights[i]
            # add back to the new model
            layer_weights.data.add_(contribution)

    # return the new model
    return new_model


def main():
    # Use argparse to get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=2, help="number of workers")

    args = parser.parse_args()

    num_workers = args.num_workers

    if num_workers > MAX_CLIENTS:
        raise ValueError(
            "Number of clients must be less than or equal to ", MAX_CLIENTS
        )

    # # wandb experiments

    # wandb_track = False  # <-needs to be exported to yaml

    # if wandb_track:
    #     init_wandb(num_workers)

    # Load data using the data distributor class
    data_distributor = None
    if data_distributor is None:
        data_distributor = DATA_DISTRIBUTOR(num_workers)
        trainloaders = data_distributor.get_trainloaders()
        test_data = data_distributor.get_testloader()

    # Shut down Ray if it has already been initialized
    if ray.is_initialized():
        ray.shutdown()
    # Instantiate the server and models
    ray_ctx = ray.init(namespace="federated_learning", num_cpus=num_workers + 1)

    print("============================== INFO ==============================")
    main_node_address = ray_ctx.address_info["redis_address"]
    print(f"Ray initialized with address: {main_node_address}")
    cluster_resources = ray.cluster_resources()
    print(f"Ray initialized with resources: {cluster_resources}")
    print("============================== END ==============================")

    # Spawn worker actors

    # input_layer_size = 3072

    next_model = FedAvgCNN()
    weights = [0.5, 0.5]

    for iteration in range(N_ITERATIONS):
        workers = [
            ParticipantActor.remote(
                deepcopy(next_model), iteration + 1, trainloaders[i], test_data
            )
            for i in range(num_workers)
        ]

        # Start training on each worker
        train_futures = [worker.train.remote(EPOCHS) for worker in workers]
        train_results = ray.get(train_futures)  # Wait for training to complete

        clients_params = [params_from_model(model) for model in train_results]
        next_model = average(FedAvgCNN(), clients_params, weights=weights)

        # Start testing on each worker
        test_futures = [worker.test.remote() for worker in workers]
        test_results = ray.get(test_futures)

        # Aggregate test results
        avg_loss = sum([result[0] for result in test_results]) / len(test_results)
        avg_accuracy = sum([result[1] for result in test_results]) / len(test_results)
        print("Accuracies: ", [result[1] for result in test_results])
        print(f"Average Test Loss: {avg_loss}\nAverage Accuracy: {avg_accuracy}%")

    # plot_workers_losses(workers)

    # if wandb_track:
    #     wandb.finish()
    ray.shutdown()


if __name__ == "__main__":
    main()
