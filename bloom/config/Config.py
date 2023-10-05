import yaml
import os


class Config:
    DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "default_config.yaml")

    def __init__(self, config_file: str):
        self.config = self.parse_config_file(config_file)

    def parse_config_file(self, config_file: str):
        """
        reads the configuration from the YAML file specified
        returns the config as dictionary object

        Args:
            config_filepath: path to the YAML file containing the configuration

        """
        if not (config_file.lower().endswith((".yaml", ".yml"))):
            print("Please provide a path to a YAML file.")
            quit()
        with open(config_file, "r") as config_file:
            config = yaml.safe_load(config_file)
        return config

    def get_available_datasets(self):
        return self.config["datasets"]["available"]

    def get_chosen_datasets(self):
        return self.config["datasets"]["chosen"]

    def get_available_optimizers(self):
        return self.config["optimizers"]["available"]

    def get_chosen_optimizers(self):
        return self.config["optimizers"]["chosen"]

    def get_available_loss(self, task: str):
        return self.config["loss_functions"][task]["available"]

    def get_chosen_loss(self, task: str):
        return self.config["loss_functions"][task]["chosen"]

    def get_wand_active(self):
        return self.config["wandb"]["active_tracking"]

    def get_wandb_key(self):
        return self.config["wandb"]["login_key"]

    def get_hyperparams(self):
        return self.config["hyper-params"]
