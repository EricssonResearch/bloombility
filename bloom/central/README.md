# Centralized CNN

A simple centralized CNN for a classification task using the FEMNIST and CIFAR10 datasets, or a regression task using the CaliforniaHousing dataset.

## Running Instructions
To run the script, you need to pass the task you want to perform as a flag.
Available is either ```classification``` or ```regression```.

Run the script as follows:

```console
python main.py -t "classification"
```

All other configuration is read from a YAML file that can be provided via command line argument using the flag ```-c ``` or ```--config```. A default YAML is provided by ```config/default_config.yaml```.
Imports of other packages are organized in "context.py"

### Experiment tracking with wandb

To track experiments using wandb, set the following paramenters in "default_config.yaml":

```yaml
  active_tracking: True
  login_key: $your_login_key
```

To find your login API key:

  1. log into the website https://wandb.ai/site
  2. go to the question mark icon titled "Resource&help" in the top right corner of the screen
  3. select "Quickstart" from the dropdown menu
  4. find the yellow text box labeled "Your API key for logging in to the wandb library"

### YAML Configuration

As of now, the script expects the following configuration values:
(updated 10.10.2023)

```yaml
datasets:
  classification:
    available: list of string
    chosen: string
  regression:
    available: list of string
    chosen: string

optimizers:
  classification:
    available: list of string
    chosen: string
  regression:
    available: list of string
    chosen: string

loss_functions:
  classification:
    available: list of string
    chosen: string
  regression:
    available: list of string
    chosen: string

wandb:
  active_tracking: bool
  login_key: string

hyper-params:
  batch_size: int
  learning_rate: float
  num_epochs: int
  num_workers: int

```
