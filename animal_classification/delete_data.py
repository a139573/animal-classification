import os
import re
from pathlib import Path
"""
Script to clean dataset folders by deleting specific files.

This script iterates through subdirectories of a root data folder
(e.g., class folders like 'cat', 'dog') and deletes files
whose names match a specific regex pattern.

It is designed to remove files that may be artifacts from a renaming
process, such as "cat_1.jpg", "lion_12.png", etc.

Notes
-----
The variables `root_dir` and `pattern` must be set manually
within this script before execution.
"""

root_dir = Path("/home/alumno/Documents/software/animal-classification/data/mini_animals/animals")

# Regex to match files like "cat_1.jpg", "lion_12.png", etc.
pattern = re.compile(r"^[A-Za-z]+_[0-9]+\.(jpg|jpeg|png)$", re.IGNORECASE)

deleted = 0

for class_folder in root_dir.iterdir():
	print(f"Class folder is  {class_folder}")
	if class_folder.is_dir():
		for file in class_folder.iterdir():
			if file.is_file() and pattern.match(file.name):
				print(f"Deleting {file}")
				file.unlink()
				deleted += 1

print(f"\n✅ Done! Deleted {deleted} renamed files.")
