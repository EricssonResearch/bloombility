## Centralized CNN

A simple centralized CNN for a classification task using the FEMNIST and CIFAR10 datasets.

### Running Instructions

The configuration is read from a YAML file provided via command line argument. \
Run the script as follows:
```console
python3 simple_classification.py '/path/to/yaml/file'
```

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

wandb_tracking:
  activated: bool

hyper-params:
  batch_size: int
  learning_rate: float
  num_epochs: int
  num_workers: int
```

TODO: Provide default config if none is provided by user?

Current working configuration for regression_pytorch.py:
-   datasets: CaliforniaHousing
-   optimizers: Adam
-   loss_functions: MSELoss
