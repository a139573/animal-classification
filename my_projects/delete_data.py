import os
import re
from pathlib import Path

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
