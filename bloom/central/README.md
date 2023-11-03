# Centralized CNN

A simple centralized CNN for a classification task using the FEMNIST and CIFAR10 datasets, or a regression task using the CaliforniaHousing dataset.

## Running Instructions

```console
python main.py
```

## Configuration Management

We are using [Hydra](https://hydra.cc/) as configuration management tool. We provide a default configuration [here](/bloom/config/central/).
The default config can be overridden in two ways:

### Specify custom config files

You can specify custom config files in order to override the default config, and specify the custom config as command-line argument.

For example, if you define a custom client config called `main_custom.yaml`, you can specify as follows:

```
python main.py main=main_custom
```

:exclamation: NOTE:
- By providing a custom configuration file, the default configuration is completely overwritten. Therefore, all the expected configuration values need to be specified.
- Make sure to place the configuration files in the correct folder ([`/config/central/main/`](/bloom/config/central/main/)).

### Specify custom config values

Alternatively, you can specify single configuration as command-line argument like so:

```
python main.py main.hyper_params.learning_rate=4
```

This way, only the specified values are overwritten.

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
