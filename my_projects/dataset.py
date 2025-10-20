import os
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# # --- Ask user for random seed ---
# while True:
#     try:
#         seed = int(input("Enter a random seed (integer): "))
#         break
#     except ValueError:
#         print("Please enter a valid integer.")

# torch.manual_seed(seed)

# # --- Ask user for test size ---
# while True:
#     try:
#         test_frac = float(input("Enter test fraction (0.05 to 0.5): "))
#         if 0.05 <= test_frac <= 0.5:
#             break
#         else:
#             print("Test fraction must be between 0.05 and 0.5")
#     except ValueError:
#         print("Please enter a valid number.")

# --- Ask the user which dataset to use ---
# dataset_choice = ""
# data_dir = os.path.join(os.getcwd(), "data")
# while dataset_choice not in ["1", "2"]:
#     print("Select the dataset to use:")
#     print("1: Full dataset (animals)")
#     print("2: Reduced dataset (mini_animals)")
#     dataset_choice = input("Enter 1 or 2: ")

# if dataset_choice == "1":
#     subsample_dir = os.path.join(data_dir, "animals")
# else:
#     subsample_dir = os.path.join(data_dir, "mini_animals")

# --- Lightning DataModule ---
class AnimalsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir=".", batch_size=32, num_workers=8, test_frac=0.2, seed=123):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.test_frac = test_frac
        self.seed = seed
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

        # Compute split sizes
        n_total = len(full_dataset)
        n_test = int(n_total * self.test_frac)
        n_trainval = n_total - n_test
        n_val = int(n_trainval * 0.25)  # 25% of train+val for validation
        n_train = n_trainval - n_val

        # Use torch.Generator for reproducibility
        generator = torch.Generator().manual_seed(self.seed)

        if stage == "fit" or stage is None:
            self.train_data, self.val_data, self.test_data = random_split(
                full_dataset, [n_train, n_val, n_test], generator=generator
            )
        if stage == "validate":
            self.val_data, _, _ = random_split(
                full_dataset, [n_train, n_val, n_test], generator=generator
            )
        if stage == "test":
            _, _, self.test_data = random_split(
                full_dataset, [n_train, n_val, n_test], generator=generator
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


# # --- Instantiate DataModule ---
# data_module = AnimalsDataModule(
#     data_dir=subsample_dir, test_frac=test_frac, seed=seed
# )