# Centralized CNN

A simple centralized CNN for a classification task using the FEMNIST and CIFAR10 datasets.

## Running Instructions

The configuration is read from a YAML file provided via command line argument. \
Run the script as follows:

```console
python3 simple_classification.py '/path/to/yaml/file'
```

### Experiment tracking with wandb

To track experiments using wandb, set the following paramenters in "default_config.yaml":

```yaml

  active_tracking: True
  login_key: $your_login_key
```

To find your login API key:

  1. log into the website "https://wandb.ai/site"
  2. go to the question mark icon titled "Resource&help" in the top right corner of the screen
  3. select "Quickstart" from the dropdown menu
  4. find the yellow text box labeled "Your API key for logging in to the wandb library"

### YAML Configuration

As of now, the script expects the following configuration values:

```yaml

task:
  available: list of string (i.e. classification, regression)
  chosen: string

datasets:
  available: list of string
  chosen: string

optimizers:
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

Current working configuration for regression_pytorch.py:

- datasets: CaliforniaHousing
- optimizers: Adam
- loss_functions: MSELoss
