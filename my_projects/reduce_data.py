import os
import random
from pathlib import Path
from PIL import Image
from shutil import copy2

# Paths
data_dir = Path("data/animals")           # Original dataset
mini_dir = Path("data/mini_animals")      # Reduced dataset
mini_dir.mkdir(parents=True, exist_ok=True)

# Parameters
target_size = (224, 224)
images_per_class = 30  # Number of images to keep per class

# Process each class folder
for class_folder in data_dir.iterdir():
    if not class_folder.is_dir():
        continue

    class_name = class_folder.name
    mini_class_folder = mini_dir / class_name
    mini_class_folder.mkdir(parents=True, exist_ok=True)

    # Get all image files
    images = [img for img in class_folder.iterdir() if img.is_file()]
    
    # Stratified subsample
    if len(images) < images_per_class:
        raise ValueError(f"Class {class_name} has less than {images_per_class} images.")
    sampled_images = random.sample(images, images_per_class)

    # Resize and save
    for img_path in sampled_images:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(target_size)
        save_path = mini_class_folder / img_path.name
        img.save(save_path)

    print(f"Processed class '{class_name}': {len(sampled_images)} images saved to {mini_class_folder}")
