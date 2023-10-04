__author__ = "tao-shen"
__license__ = "MIT"
__repository__ = "https://github.com/tao-shen/FEMNIST_pytorch"

from torchvision.datasets import MNIST, utils
from PIL import Image
import os.path
import os
import torch

"""
    A class used to represent the FEMNIST dataset

    Attributes:
        resources: origin of the zipped dataset to be downloaded

"""


class FEMNIST(MNIST):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    # origin of the FEMNIST dataset
    resources = [
        (
            "https://raw.githubusercontent.com/tao-shen/FEMNIST_pytorch/master/femnist.tar.gz",
            "59c65cec646fc57fe92d27d83afdf0ed",
        )
    ]

    """Initializer of the FEMNIST class

        Args:
            train: boolean whether the dataset is supposed to be for training (true) or testing (false) purposes. Default True
            transform: transformations applied to the dataset. Default None
            download: Whether files are supposed to be downloaded if not yet present. Default False
        Returns:
            FEMNIST object
        Raises:
            RunTimeError: If dataset not found but download == False
    """

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super(MNIST, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.train = train

        # no files there, user wants to download
        if download and not self.check_files_exist():
            print("Downloading...")
            print("Folders exist:", self.check_folders_exist())
            self.download()
        # files already exists, user want to download: do not download
        elif download and self.check_files_exist():
            print("Files already exist, processing...")
        # no download, but files do not exist
        elif not download and not self.check_files_exist():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.data, self.targets, self.users_index = torch.load(
            os.path.join(self.processed_folder, data_file)
        )

    """Get image and target label at given index
        Args:
            index: index of image
        Returns:
            img: Image
            target: target label

    """

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="F")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    """ Checks whether expected files exist

        Returns:
            a boolean value that indicates whether all files were found
    """

    def check_files_exist(self):
        processed_folder = os.path.join(self.root, self.__class__.__name__, "processed")
        test_file = "test.pt"
        train_file = "training.pt"
        exists = True
        for file in [test_file, train_file]:
            cur = os.path.join(processed_folder, file)
            found = os.path.isfile(cur)
            if not found:
                exists = False

        return exists

    """ Checks whether expected folders exist

        Returns:
            a boolean value that indicates whether all folders were found
    """

    def check_folders_exist(self):
        raw_folder = os.path.join(self.root, self.__class__.__name__, "raw")
        processed_folder = os.path.join(self.root, self.__class__.__name__, "processed")
        if os.path.isdir(raw_folder) and os.path.isdir(processed_folder):
            return True
        else:
            return False

    """Download the FEMNIST data if it doesn't exist in processed_folder already.
    """

    def download(self):
        import shutil

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        # download files
        for url, md5 in self.resources:
            filename = url.rpartition("/")[2]
            utils.download_and_extract_archive(
                url, download_root=self.raw_folder, filename=filename, md5=md5
            )

        # new_processed = os.path.join(self.root, self.__class__.__name__, "processed")
        # print("directory:", new_processed)

        # process and save as torch files
        print("Processing...")
        shutil.move(
            os.path.join(self.raw_folder, self.training_file), self.processed_folder
        )
        shutil.move(
            os.path.join(self.raw_folder, self.test_file), self.processed_folder
        )
        print("Processing done")
