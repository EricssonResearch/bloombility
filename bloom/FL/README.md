This directory contains code for Federated Learning.

## Runtime instructions

Before running, make sure to follow the steps described in [CONTRIBUTING.md](/CONTRIBUTING.md) in order to set up a virtual environment and install all the necessary dependencies.

### Run locally
The entry point is `main.py`, which splits the datasets (using the [data distributor](/bloom/load_data/)), launches a server and runs a Flower simulation with a specified number of clients.\
To run the simulation, simply run `python main.py`.


## Folder Structure

```
FL
│   README.md
│   main.py           # entry point for the Flower simulation
│
└───client
│   │   client.py     # Flower client along with train and test functions
│
└───server
    │   server.py     # Flower server that defines the FL strategy and starts the Flower simulation
    │   utils.py      # utility python file containing FL strategies and helper functions
```

## Configuration Management

We are using [Hydra](https://hydra.cc/) as configuration management tool. We provide default configs for main, the client, and server [here](/bloom/config/federated/).
The default config can be overridden in two ways:

### Specify custom config files

You can specify custom config files in order to override the default config, and specify the custom config as command-line argument.

For example, if you define a custom client config called `client_custom.yaml`, you can specify as follows:

```
python main.py client=client_custom
```

:exclamation: NOTE:
- By providing a custom configuration file, the default configuration is completely overwritten. Therefore, all the expected configuration values need to be specified.
- Make sure to place the configuration files in the correct folder (i.e., `main/`, `client/`, or `server/`).

### Specify custom config values

Alternatively, you can specify single configuration as command-line argument like so:

```
python main.py client.hyper_params.learning_rate=4 server.num_rounds=6
```

This way, only the specified values are overwritten.
