import shutil
from pathlib import Path
import kagglehub

"""
Dataset Preparation Script.

Downloads the "Animal Image Dataset" from Kaggle, flattens nested directories,
and ensures class folders are directly under './data/animals'.
"""

def flatten_folder(path: Path) -> Path:
    """
    Recursively flattens a directory structure.

    If a folder contains exactly one subfolder and no files,
    this function descends into that subfolder repeatedly.
    """
    while True:
        items = list(path.iterdir())
        dirs = [i for i in items if i.is_dir()]
        files = [i for i in items if i.is_file()]

        if len(dirs) == 1 and len(files) == 0:
            path = dirs[0]
        else:
            break
    return path

def main():
    """
    Download, flatten, and copy dataset to 'data/animals'.
    """
    target_dir = Path("data/animals")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    dataset_path = Path(kagglehub.dataset_download(
        "iamsouravbanerjee/animal-image-dataset-90-different-animals"
    ))
    print(f"Downloaded to: {dataset_path}")

    # First flatten
    dataset_path = flatten_folder(dataset_path)

    # Extra flatten if top-level folder is still 'animals'
    subdirs = [c for c in dataset_path.iterdir() if c.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name.lower() == "animals":
        dataset_path = subdirs[0]
        print(f"Detected nested 'animals/' folder, descending into it...")

    print("Copying files to target directory...")
    for item in dataset_path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            if dest.exists():
                print(f"Skipping existing folder: {dest}")
            else:
                shutil.copytree(item, dest)
                print(f"Copied folder: {item.name}")
        else:
            shutil.copy2(item, dest)
            print(f"Copied file: {item.name}")

    print(f"\n✅ Dataset ready under: {target_dir.resolve()}")

if __name__ == "__main__":
    main()
