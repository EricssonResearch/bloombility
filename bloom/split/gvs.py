from gvslearning.utils.util import get_dataset
from gvslearning.data.dataloader import get_data_loaders
from bloom import ROOT_DIR
import os

args = {
    "data": {
        # "train-data": "`{Root}/bloom/data/telecom/train.pkl`",
        "train-data": os.path.join(ROOT_DIR, "split", "data", "telecom", "train.pkl"),
        "eval-data": os.path.join(ROOT_DIR, "split", "data", "telecom", "validate.pkl"),
        "test-data": os.path.join(ROOT_DIR, "split", "data", "telecom", "test.pkl"),
    },
    "period": 24,
    "output-size": 6,
}

dataset, validate_dataset, test_dataset = get_dataset(args)


batch_size = 8
train_loader, eval_loader, test_loader = get_data_loaders(
    [dataset, validate_dataset, test_dataset], batch_size
)
first_batch = next(iter(train_loader))
first_batch_data, first_batch_label = first_batch
# print("train_loader: ", train_loader)
print("Length of train_loader: ", len(train_loader))
print("Data of First batch of train_loader: ", first_batch_data)
print("Label of first batch of train_loader: ", first_batch_label)
