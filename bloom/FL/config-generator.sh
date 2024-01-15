#!/bin/bash

# Define the parameters
learning_rates=(0.01 0.001 0.005)
num_clients=(2 4 8 16)
batch_sizes=(16 32 64)

# Iterate over all combinations of parameters
for lr in ${learning_rates[@]}; do
  for nc in ${num_clients[@]}; do
    for bs in ${batch_sizes[@]}; do
      # Define the filename
      filename="config_lr_${lr}_nc_${nc}_bs_${bs}.yaml"

      # Create the yaml file
      cat > $filename << EOF
main:
  # available: FEMNIST, CIFAR10
  dataset: "CIFAR10"
client:
  num_clients: $nc
  hyper_params:
      learning_rate: $lr
      num_epochs: 1
      batch_size: $bs
server:
  num_rounds: 5
  # available: FedAvg, FedAdam, FedAvgM. FedYogi, FedAdagrad, CustomFedAvg
  strategy: 'FedAvg'
  wandb_active: False
  wandb_key: ""
EOF
    done
  done
done
