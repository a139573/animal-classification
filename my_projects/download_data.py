"""
download_dataset.py
--------------------
Downloads the Kaggle dataset "Animal Image Dataset (90 Different Animals)"
using kagglehub and extracts it into data/animals/ preserving the subfolders.
"""

import os
import shutil
from pathlib import Path
import kagglehub

def main():
    # --- Target directory ---
    target_dir = Path("data/animals")
    target_dir.mkdir(parents=True, exist_ok=True)

    # --- Download dataset via kagglehub ---
    print("Downloading dataset from Kaggle...")
    dataset_path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")
    dataset_path = Path(dataset_path)
    print(f"Downloaded to: {dataset_path}")

    # --- Copy dataset contents to data/animals/ ---
    print("Copying files to data/animals/ ...")
    for item in dataset_path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            # Copy the entire folder
            if dest.exists():
                print(f"Skipping existing folder: {dest}")
            else:
                shutil.copytree(item, dest)
                print(f"Copied folder: {item.name}")
        else:
            # Copy individual files if needed
            shutil.copy2(item, dest)
            print(f"Copied file: {item.name}")

    print(f"\n✅ Dataset ready under: {target_dir.resolve()}")

if __name__ == "__main__":
    main()