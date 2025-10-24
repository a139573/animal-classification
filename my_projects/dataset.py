import os
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class AnimalsDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for an animal classification dataset.

    This class handles the loading, transformation, and splitting (training,
    validation, test) of the image dataset.

    Attributes
    ----------
    data_dir : str
        Path to the data directory.
    batch_size : int
        Number of samples per batch.
    num_workers : int
        Number of subprocesses for data loading.
    transform : torchvision.transforms.Compose
        Transformations applied to each image.
    class_names : list[str]
        Names of the dataset classes. Defined in `setup()`.
    train_data : torch.utils.data.Dataset
        Training data subset.
    val_data : torch.utils.data.Dataset
        Validation data subset.
    test_data : torch.utils.data.Dataset
        Test data subset.
    """
    def __init__(self, data_dir=".", batch_size=32, num_workers=4, test_frac=0.2, seed=123):
        """
        Initializes the DataModule.

        Parameters
        ----------
        data_dir : str, optional
            Directory containing the dataset (default is ".").
        batch_size : int, optional
            Batch size for the DataLoaders (default is 32).
        num_workers : int, optional
            Number of workers for the DataLoaders (default is 4).
        test_frac : float, optional
            Fraction of the dataset to reserve for testing (default is 0.2).
        seed : int, optional
            Random seed for data splitting (default is 123).
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_frac = test_frac
        self.seed = seed
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def setup(self, stage=None):
        """
        Prepares the data and performs the train/val/test split.

        This method is called automatically by PyTorch Lightning. It loads the
        full dataset, calculates the splits, and assigns the
        `train_data`, `val_data`, and `test_data` subsets.

        Parameters
        ----------
        stage : str, optional
            Lightning stage (e.g., "fit", "test"). Not used in this
            implementation, as the data is split only once.
        """
        if not hasattr(self, "train_data"):
            full_dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
            self.class_names = full_dataset.classes

            n_total = len(full_dataset)
            n_test = int(n_total * self.test_frac)
            n_val = int((n_total - n_test) * 0.25)
            n_train = n_total - n_val - n_test

            generator = torch.Generator().manual_seed(self.seed)
            self.train_data, self.val_data, self.test_data = random_split(
                full_dataset, [n_train, n_val, n_test], generator=generator
            )

    def train_dataloader(self):
        """
        Creates the DataLoader for the training set.

        Returns
        -------
        torch.utils.data.DataLoader
            The training DataLoader, with shuffling (shuffle=True).
        """
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        """
        Creates the DataLoader for the validation set.

        Returns
        -------
        torch.utils.data.DataLoader
            The validation DataLoader (shuffle=False).
        """
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Creates the DataLoader for the test set.

        Returns
        -------
        torch.utils.data.DataLoader
            The test DataLoader (shuffle=False).
        """
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)



# # --- Instantiate DataModule ---
# data_module = AnimalsDataModule(
#     data_dir=subsample_dir, test_frac=test_frac, seed=seed
# )
