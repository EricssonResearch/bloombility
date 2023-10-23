This directory contains code for Federated Learning.

## Runtime instructions

Before running, make sure to follow the steps described in [CONTRIBUTING.md](/CONTRIBUTING.md) in order to set up a virtual environment and install all the necessary dependencies.

### Run locally
The entry point is `main.py`, which splits the datasets (using the [data distributor](/bloom/load_data/)), launches a server and runs a Flower simulation with a specified number of clients.\
To run the simulation, simply run `python main.py`.

As of now, the configuration is fixed. In future, a configuration manager will be added.

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
