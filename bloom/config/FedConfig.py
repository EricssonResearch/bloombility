import yaml
import os
from Config import Config
from definitions import ROOT_DIR


class FedConfig(Config):
    DEFAULT_CONFIG = os.path.join(ROOT_DIR, "config", "default_config_FL.yaml")

    def __init__(self, config_file: str):
        super().__init__(config_file)
