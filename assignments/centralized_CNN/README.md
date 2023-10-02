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
datasets:
  available: list of string
  chosen: string

optimizers:
  available: list of string
  chosen: string

classification:
  loss_functions:
    available: list of string
    chosen: string

hyper-params:
  batch_size: int
  learning_rate: float
  num_epochs: int
  num_workers: int
```