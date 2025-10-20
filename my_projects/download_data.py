import shutil
from pathlib import Path
import kagglehub

def flatten_folder(path: Path) -> Path:
    """
    If the folder contains exactly one folder and no files,
    return that subfolder instead (flattening).
    Repeat recursively until we reach the actual content.
    """
    while True:
        items = list(path.iterdir())
        dirs = [i for i in items if i.is_dir()]
        files = [i for i in items if i.is_file()]

        # If only one folder and no files, descend into it
        if len(dirs) == 1 and len(files) == 0:
            path = dirs[0]
        else:
            break
    return path

def main():
    target_dir = Path("data/animals")
    target_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset from Kaggle...")
    dataset_path = Path(kagglehub.dataset_download(
        "iamsouravbanerjee/animal-image-dataset-90-different-animals"
    ))
    print(f"Downloaded to: {dataset_path}")

    # Flatten redundant top-level folders
    dataset_path = flatten_folder(dataset_path)

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