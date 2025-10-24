import os
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def reduce_dataset(data_dir, output_dir, images_per_class, image_size, progress=None, seed=123):
    """Reduce dataset by sampling a fixed subset of images per class and resizing them."""
    random.seed(seed)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # 🧹 clear previous output to prevent duplicates
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [c for c in data_dir.iterdir() if c.is_dir()]
    total = len(classes)

    for i, class_folder in enumerate(classes):
        class_name = class_folder.name
        mini_class_folder = output_dir / class_name
        mini_class_folder.mkdir(parents=True, exist_ok=True)

        images = sorted([img for img in class_folder.iterdir() if img.is_file()])
        if len(images) < images_per_class:
            raise ValueError(f"Class '{class_name}' has only {len(images)} images (<{images_per_class}).")

        sampled_images = random.sample(images, images_per_class)

        for j, img_path in enumerate(tqdm(sampled_images, desc=f"Processing {class_name}")):
            if progress is not None:
                progress((i + j / len(sampled_images)) / total, desc=f"{class_name}: {j+1}/{len(sampled_images)} images")
            with Image.open(img_path).convert("RGB") as img:
                img = img.resize((image_size, image_size))
                img.save(mini_class_folder / img_path.name)

        print(f"✅ Processed class '{class_name}': {len(sampled_images)} images saved to {mini_class_folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce animal dataset by sampling and resizing images.")
    parser.add_argument(
        "--images", type=int, choices=[15, 30, 60], default=30,
        help="Number of images per class (15, 30, or 60). Default: 30"
    )
    parser.add_argument(
        "--size", type=int, choices=[56, 112, 224], default=224,
        help="Target image size (56, 112, or 224). Default: 224"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/animals/animals",
        help="Path to the original dataset directory."
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/mini_animals/animals",
        help="Path to save the reduced dataset."
    )

    args = parser.parse_args()

    reduce_dataset(args.data_dir, args.output_dir, args.images, args.size)
