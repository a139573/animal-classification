"""
# Preprocessing & Data Management

This module handles the preparation of data before it enters the model. 
It includes the PyTorch Lightning DataModule for training and scripts for 
managing the raw dataset.

## Components
- **AnimalsDataModule**: The main class that handles loading, splitting, and transforming images.

## Scripts
This folder also contains standalone scripts for data management:
- `download_data.py`: Downloads the raw images.
- `reduce_data.py`: Resizes huge images to save space.
- `delete_data.py`: Cleans up the dataset.
"""

from .dataset import AnimalsDataModule

# Only export the reusable class. 
# We do not import the scripts to prevent them from running accidentally.
__all__ = [
    "download_data",
    "reduce_data",
    "delete_data",
    "dataset"]