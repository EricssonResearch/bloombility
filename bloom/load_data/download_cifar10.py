from torchvision.datasets import CIFAR10, utils
import torchvision.transforms as transforms
import torch
import os


class CIFARTEN(CIFAR10):
    """
    A class used to represent and handle the CIFAR10 dataset

    """

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        """Initializer of the CIFAR10 class

        Args:
            train: boolean whether the dataset is supposed to be for training (true) or testing (false) purposes. Default True
            transform: transformations applied to the dataset. Default None
            download: Whether files are supposed to be downloaded if not yet present. Default False
        Returns:
            CIFAR10 object
        Raises:
            RunTimeError: If dataset not found but download == False
        """
        super(CIFAR10, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.train = train

        # no files there, user wants to download
        if download and not self.check_files_exist():
            print("Downloading...")
            print("Folders exist:", self.check_folders_exist())
            super().download()
        # files already exists, user want to download: do not download
        elif download and self.check_files_exist():
            print("Files already exist, processing...")
        # no download, but files do not exist
        elif not download and not self.check_files_exist():
            raise RuntimeError(
                "Dataset not found." + " You can use download=True to download it"
            )

        return

    def check_files_exist(self):
        """Checks whether expected files exist

        Returns:
            a boolean value that indicates whether all files were found
        """
        data_files = [
            "data_batch_1",
            "data_batch_2",
            "data_batch_3",
            "data_batch_4",
            "data_batch_5",
            "test_batch",
        ]
        self.data_folder = os.path.join(self.root, "cifar-10-batches-py")
        exists = True
        for file in data_files:
            cur = os.path.join(self.data_folder, file)
            found = os.path.isfile(cur)
            if not found:
                exists = False

        return exists

    def check_folders_exist(self):
        """Checks whether expected folders exist

        Returns:
            a boolean value that indicates whether all folders were found
        """
        self.data_folder = os.path.join(self.root, "cifar-10-batches-py")
        if os.path.isdir(self.data_folder):
            return True
        else:
            return False

    def get_cifar10_datasets(root, transform):
        """Downloads the train and test sets of CIFAR10

        Returns:
            the train-set and the test-set
        """

        trainset = CIFARTEN(root=root, train=True, download=True, transform=transform)

        testset = CIFARTEN(root=root, train=False, download=True, transform=transform)

        return trainset, testset


"""
Example of using this class:

if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset, testset = CIFARTEN.get_cifar10_datasets('./data',transform)
"""
