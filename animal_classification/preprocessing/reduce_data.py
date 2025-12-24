"""
Dataset Reduction Utility.

Reduces an image dataset by sampling a fixed number of images per class
and resizing them to a uniform size. Saves reduced dataset to ../data/mini_animals/animals
"""
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def reduce_dataset(images_per_class, image_size, progress=None, seed=123):
    """
    Reduces the animal dataset by sampling and resizing images.

    It clears the output directory if it exists and repopulates it with
    the sampled images.

    Parameters
    ----------
    images_per_class : int.
        Number of images to randomly select and save for each class.
        
    image_size : int.
        The target width and height (in pixels) to resize images to.

    progress : callable, optional.
        A callback function to update progress bars (e.g., for Gradio).
        Should accept a float (0.0 to 1.0) and a description string.
        
    seed : int, optional.
        Random seed for reproducibility (default is 123).
    """
    random.seed(seed)

    # Use the current working directory
    project_root = Path.cwd()
    
    # Point to the location from download_data.py
    data_dir = project_root / "data" / "animals" / "animals"
    # Output location"
    output_dir = project_root / "animal_classification" / "data" / "mini_animals" / "animals"

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_dir}. Did you run download_data.py?")

    # Clear old reduced dataset
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = [c for c in data_dir.iterdir() if c.is_dir()]
    total = len(classes)

    if total == 0:
         raise ValueError(f"No class folders found in {data_dir}")

    for i, class_folder in enumerate(classes):
        class_name = class_folder.name
        mini_class_folder = output_dir / class_name
        mini_class_folder.mkdir(parents=True, exist_ok=True)

        images = sorted([img for img in class_folder.iterdir() if img.is_file()])
        
        # Safety check: if a class has fewer images than requested, use what is available
        # (or raise error if you want strict enforcement)
        if len(images) < images_per_class:
            print(f"Warning: Class '{class_name}' has only {len(images)} images. Using all of them.")
            sampled_images = images
        else:
            sampled_images = random.sample(images, images_per_class)

        for j, img_path in enumerate(tqdm(sampled_images, desc=f"Processing {class_name}")):
            # Update progress callback if provided (useful for Gradio integration later)
            if progress is not None:
                progress((i + j / len(sampled_images)) / total,
                         desc=f"{class_name}: {j+1}/{len(sampled_images)} images")
            
            try:
                with Image.open(img_path).convert("RGB") as img:
                    img = img.resize((image_size, image_size))
                    img.save(mini_class_folder / img_path.name)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")

        print(f"✅ Processed class '{class_name}': {len(sampled_images)} images saved to {mini_class_folder}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Reduce animal dataset by sampling and resizing images.")
    parser.add_argument(
        "--num-images", type=int, choices=[15, 30, 60], default=30,
        help="Number of images per class (15, 30, or 60). Default: 30"
    )
    parser.add_argument(
        "--img-size", type=int, choices=[56, 112, 224], default=224,
        help="Target image size (56, 112, or 224). Default: 224"
    )

    args = parser.parse_args()

    reduce_dataset(args.num_images, args.img_size)


if __name__ == "__main__":
    main()