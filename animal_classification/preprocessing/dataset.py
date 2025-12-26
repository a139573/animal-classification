"""
# 🖼️ Animals Data Module

This module handles the loading and preprocessing of the animal images.
It uses PyTorch Lightning's `DataModule` to organize:
1.  **Transformations:** Mandatory normalization and Data Augmentation for training.
2.  **Splitting:** Dividing the dataset into training and validation sets.
3.  **Dataloaders:** Batching and shuffling for CPU/GPU processing.
"""

from pathlib import Path
from typing import Optional

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets

class AnimalsDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the Animal Image Dataset.

    Ensures that images are augmented during training and normalized 
    correctly for pre-trained VGG architectures.
    """
    def __init__(self, data_dir: str, batch_size: int = 16, seed: int = 42, test_frac: float = 0.2, num_workers: int = 2):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seed = seed
        self.test_frac = test_frac
        self.num_workers = num_workers
        
        # 1. Training Transformations (Data Augmentation)
        # Random cropping and flipping are essential to prevent overfitting on small datasets
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

        # 2. Validation/Test Transformations (Deterministic)
        # Consistent with ImageNet validation protocols
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])

    def setup(self, stage: Optional[str] = None):
        """
        Loads the dataset and performs the train/validation split using 
        distinct transformation pipelines for each subset.
        """
        # We create two instances of the dataset pointing to the same root
        # but with different transformations.
        train_full = datasets.ImageFolder(root=self.data_dir, transform=self.train_transform)
        val_full = datasets.ImageFolder(root=self.data_dir, transform=self.val_transform)
        
        self.class_names = train_full.classes
        
        # Logic to ensure the same indices are used for splitting despite having two objects
        num_samples = len(train_full)
        indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(self.seed)).tolist()
        
        n_val = int(num_samples * self.test_frac)
        
        # Create subsets that share the underlying data but apply the correct transform
        self.train_ds = Subset(train_full, indices[n_val:])
        self.val_ds = Subset(val_full, indices[:n_val])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )