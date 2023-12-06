import wandb
import torch

from bloom.load_data.data_distributor import DATA_DISTRIBUTOR
from client import generate_client_fn
from server import FlowerServer
from bloom import ROOT_DIR, models

import os
import glob
import hydra
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

config_path = os.path.join(ROOT_DIR, "config", "federated")
DEVICE = torch.device("cpu")


def eval_final(model, test_loader):
    """visualize performance of the model"""

    actuals, predictions = test_label_predictions(model, test_loader)
    print("F1 score: %f" % f1_score(actuals, predictions, average="micro"))
    print("Confusion matrix:")
    cm = confusion_matrix(actuals, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


def test_label_predictions(model, test_loader):
    """final test to visualize performance of the model"""
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            prediction = output.argmax(dim=1, keepdim=True)
            actuals.extend(target.view_as(prediction))
            predictions.extend(prediction)
    return [i.item() for i in actuals], [i.item() for i in predictions]


@hydra.main(config_path=config_path, config_name="base", version_base=None)
def main(cfg: DictConfig):
    print(config_path)
    print(OmegaConf.to_yaml(cfg))
    # PARAMS
    # Number of rounds of federated learning
    n_rounds = cfg.server.num_rounds

    # Strategies available:  ["FedAvg", "FedAdam", "FedYogi", "FedAdagrad", "FedAvgM"]
    strategy = cfg.server.strategy
    batch_size = cfg.client.hyper_params.batch_size
    num_epochs = cfg.client.hyper_params.num_epochs

    num_clients = cfg.main.num_clients

    data_split = cfg.main.data_split_chosen
    data_split_config = cfg.main.data_split_config

    advanced_visualization = cfg.main.advanced_visualization

    data_distributor = DATA_DISTRIBUTOR(num_clients, data_split_config, data_split)
    # wandb experiments
    wandb_track = cfg.main.wandb_active
    wandb_key = cfg.main.wandb_key

    if wandb_track and wandb.run is None:
        wandb.login(anonymous="never", key=wandb_key)
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            entity="cs_team_b",
            project="non_iid_client_reporting_fn",
            # track hyperparameters and run metadata
            config={
                "method": "federated",
                "n_rounds": n_rounds,
                "strategy": strategy,
                "clients": num_clients,
            },
        )

    trainloaders = data_distributor.get_trainloaders()
    print("Amount of loaders:", len(trainloaders))

    for trainloader in trainloaders:
        print("Len of loader: ", len(trainloader) * batch_size)
        if wandb_track:
            wandb.log({"trainloader_len": len(trainloader) * batch_size})

    testloader = data_distributor.get_testloader()

    client_fn = generate_client_fn(trainloaders, testloader, batch_size, num_epochs)

    server = FlowerServer(strategy=strategy, num_rounds=n_rounds)
    history = server.start_simulation(
        client_fn, cfg.main.num_clients, cfg.client.num_cpu, cfg.client.num_gpu
    )

    if wandb_track:
        wandb.finish()

    print(history)

    if advanced_visualization:
        # for advanced evaluation: load the best model
        exp_folder = os.path.join(ROOT_DIR, "FL", "saved_fl_models")
        list_of_files = [fname for fname in glob.glob(f"{exp_folder}/model_round_*")]
        latest_round_file = max(list_of_files, key=os.path.getctime)
        print("Loading pre-trained model from: ", latest_round_file)
        state_dict = torch.load(latest_round_file)
        net = models.FedAvgCNN()
        net.load_state_dict(state_dict)

        eval_final(net, testloader)


if __name__ == "__main__":
    main()
