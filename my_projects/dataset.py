# Load the dataset

import os
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

torch.manual_seed(123)

project_dir = Path(os.getcwd()).parent.absolute()
data_dir = os.path.join(os.getcwd(), "data")

# Ask the user which dataset to use
dataset_choice = ""
while dataset_choice not in ["1", "2"]:
    print("Select the dataset to use:")
    print("1: Full dataset (animals)")
    print("2: Reduced dataset (mini_animals)")
    dataset_choice = input("Enter 1 or 2: ")

if dataset_choice == "1":
    subsample_dir = os.path.join(data_dir, "animals")
else:
    subsample_dir = os.path.join(data_dir, "mini_animals")


class AnimalsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=".", batch_size=32, num_workers=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def setup(self, stage=None):
        full_dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        self.class_names = full_dataset.classes

        train_size = int(len(full_dataset) * 0.6)
        val_size = int(len(full_dataset) * 0.2)
        test_size = len(full_dataset) - train_size - val_size

        if stage == "fit" or stage is None:
            self.train_data, self.val_data, _ = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        if stage == "validate" or stage is None:
            _, self.val_data, _ = random_split(
                full_dataset, [train_size, val_size, test_size]
            )
        if stage == "test" or stage is None:
            _, _, self.test_data = random_split(
                full_dataset, [train_size, val_size, test_size]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )


data_module = AnimalsDataModule(data_dir=subsample_dir)
