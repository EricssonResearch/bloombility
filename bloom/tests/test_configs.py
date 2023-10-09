from bloom import config


def check_chosen_in_available(conf_file):
    """check that chosen options are actually available (for that task)"""

    # regresssion
    assert config.Config.get_chosen_datasets(
        conf_file, "regression"
    ) in config.Config.get_available_datasets(
        conf_file, "regression"
    ), "chosen dataset for regression is not available"

    assert config.Config.get_chosen_optimizers(
        conf_file, "regression"
    ) in config.Config.get_available_optimizers(
        conf_file, "regression"
    ), "chosen optimizer for regression is not available"

    assert config.Config.get_chosen_loss(
        conf_file, "regression"
    ) in config.Config.get_available_loss(
        conf_file, "regression"
    ), "chosen loss for regression is not available"

    # classification
    assert config.Config.get_chosen_datasets(
        conf_file, "classification"
    ) in config.Config.get_available_datasets(
        conf_file, "classification"
    ), "chosen dataset for classification is not available"

    assert config.Config.get_chosen_optimizers(
        conf_file, "classification"
    ) in config.Config.get_available_optimizers(
        conf_file, "classification"
    ), "chosen optimizer for classification is not available"

    assert config.Config.get_chosen_loss(
        conf_file, "classification"
    ) in config.Config.get_available_loss(
        conf_file, "classification"
    ), "chosen loss for classification is not available"


def check_hyperparams(conf_file):
    """Check that necessary hyperparams are defined"""
    hyperparams = config.Config.get_hyperparams(conf_file)

    assert (
        hyperparams["learning_rate"] is not None
        and hyperparams["learning_rate"] <= 1
        and hyperparams["learning_rate"] >= 1e-10
    ), "you sure about that?"

    assert hyperparams["batch_size"] is not None

    assert (
        hyperparams["num_workers"] is not None
        and hyperparams["num_workers"] >= 1
        and hyperparams["num_workers"] <= 32
    ), "you sure about that?"

    assert (
        hyperparams["num_epochs"] is not None
        and hyperparams["num_epochs"] >= 1
        and hyperparams["num_epochs"] <= 250
    ), "you sure about that?"


def wandb_config(conf_file):
    """check that the wandb key is set up right"""
    wandb_key = config.Config.get_wandb_key(conf_file)
    assert wandb_key is not None, "put wandb API key in yaml file if you want tracking"
    assert isinstance(
        wandb_key, str
    ), "wrong format of wandb API key in yaml file: should be str"
    assert (
        len(wandb_key) == 40
    ), "wrong format of wandb API key in yaml file: should be 40 chars long"


def main():
    """run asserts for config file"""

    actual_config = config.Config(config.Config.DEFAULT_CONFIG)

    if config.Config.get_wand_active(actual_config):
        wandb_config(actual_config)

    check_chosen_in_available(actual_config)
    check_hyperparams(actual_config)


if __name__ == "__main__":
    main()
