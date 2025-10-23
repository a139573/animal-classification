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
    def __init__(self, data_dir=".", batch_size=32, num_workers=4, test_frac=0.2, seed=123):
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
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)



# # --- Instantiate DataModule ---
# data_module = AnimalsDataModule(
#     data_dir=subsample_dir, test_frac=test_frac, seed=seed
# )
