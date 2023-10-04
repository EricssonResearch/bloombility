#!/usr/bin/env python3

import flwr as fl

n_rounds = 3


def weighted_average(metrics):
    acc = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(acc) / sum(examples)}


print("Setting up flower server...")
fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=n_rounds),
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
    ),
)
