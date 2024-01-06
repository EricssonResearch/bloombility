# Split Learning Module

This module is part of the Bloom project and is responsible for implementing Split Learning. Split Learning is a distributed machine learning approach where the model training is split across multiple devices or nodes. Each node only has access to a portion of the model, and the nodes communicate with each other to train the model. In this implementation, worker nodes train the head of the model, and the server node trains the tail end of the model. Worker nodes train sequentially, then send their model weights to the next worker node.


## Modules
The Split Learning module consists of the following modules:

- `bloom/split/main.py`: The main entry point for the Split Learning module. Responsible for configuring and starting the training process.
- `bloom/split/server`: Server node that contains the tail end of the model.
- `bloom/split/client`: Worker/client nodes that contain the head of the model.

## Available Models

Currently, the Split Learning module supports the following models:

- Convolutional Neural Network (CNN)

The models are trained on the following datasets:

- CIFAR10
- FEMNIST

## How to Run

To run the Split Learning module, use the following command:

\`\`\`sh
python bloom/split/main.py --num-workers 2
\`\`\`

This command will start the split learning process with 2 worker nodes.

## Configuration

Alternatively, you can configure the Split Learning module by modifying the `bloom/config/split/base.yaml` file. Here are the available configuration options:

- `n_epochs`: The number of training epochs.
- `n_workers`: The number of worker nodes.
- `wandb`: Configuration for Weights & Biases integration.
  - `track`: Whether to track the training process with Weights & Biases.
  - `key`: Your Weights & Biases API key.
- `max_clients`: The maximum number of clients that can connect to the server.

## Output

The output of the training process will be saved in the `bloom/split/outputs` directory. Each training run will have its own directory named with the date and time of the run. Inside this directory, you'll find the `.hydra/config.yaml` file with the configuration used for the run.

## Contributing

Please see the main CONTRIBUTING.md file for guidelines on how to contribute.
