import argparse
import os
import classification
import regression
import centralized
import yaml

from bloom import config
from bloom import ROOT_DIR

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

config_path = os.path.join(ROOT_DIR, "config", "central")


@hydra.main(config_path=config_path, config_name="base", version_base=None)
def main(cfg: DictConfig):
    centralized.main(cfg)


if __name__ == "__main__":
    main()
