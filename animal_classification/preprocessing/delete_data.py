"""
Script to completely remove dataset folders.

This script iterates through the root data folder and deletes
all subdirectories (e.g., 'cat', 'dog') entirely.
"""

import argparse
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Delete all animal folders.")
    parser.add_argument(
        "--target-dir", 
        type=str, 
        # CAREFUL: This defaults to the mini dataset. Change to 'data/animals' to delete raw data.
        default="animal_classification/data/mini_animals/animals",
        help="Relative path to the dataset folder to delete content from."
    )
    args = parser.parse_args()

    root_dir = Path.cwd() / args.target_dir

    if not root_dir.exists():
        print(f"❌ Error: Directory not found at {root_dir}")
        return

    print(f"⚠️  WARNING: About to delete all folders inside: {root_dir}")
    confirm = input("Are you sure? (y/n): ")
    
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return

    deleted_count = 0

    # Iterate over everything in the directory
    for item in root_dir.iterdir():
        if item.is_dir():
            try:
                # shutil.rmtree removes the folder and everything inside it
                shutil.rmtree(item)
                print(f"Deleted folder: {item.name}")
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {item.name}: {e}")

    print(f"\n✅ Done! Deleted {deleted_count} folders.")

if __name__ == "__main__":
    main()