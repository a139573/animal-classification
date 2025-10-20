import torch
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from my_projects.dataset import AnimalsDataModule
from my_projects.modeling.train import VGGNet
import os


def run_inference(model_path: Path, data_dir: Path, architecture: str, output_path: Path):
    """
    Run inference using a trained VGG model (VGG11 or VGG16).

    Parameters
    ----------
    model_path : Path
        Path to the trained model state_dict (.pth file).
    data_dir : Path
        Path to the dataset directory.
    architecture : str
        Model architecture, "vgg16" or "vgg11".
    output_path : Path
        Path to save predictions and labels as .npy files.
    """
    print(f"\n🔍 Loading model: {model_path}")
    print(f"📁 Using dataset: {data_dir}")
    print(f"🧠 Architecture: {architecture}")

    # --- Setup data module ---
    data_module = AnimalsDataModule(data_dir=data_dir, batch_size=16)
    data_module.setup()
    val_loader = data_module.val_dataloader()
    num_classes = len(data_module.class_names)

    # --- Load model ---
    model = VGGNet(architecture=architecture, num_classes=num_classes, pretrained=False)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # --- Use GPU if available ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"⚙️  Running on: {device}")

    all_probs, all_labels = [], []

    # --- Inference loop ---
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            probs = F.softmax(out, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # --- Save to disk ---
    np.save(output_path / f"{architecture}_val_probs.npy", all_probs)
    np.save(output_path / f"{architecture}_val_labels.npy", all_labels)

    print(f"\n✅ Saved validation predictions to: {output_path}")


def main():
    # --- Choose dataset ---
    dataset_choice = ""
    data_dir = os.path.join(os.getcwd(), "data")
    while dataset_choice not in ["1", "2"]:
        print("Select the dataset to use:")
        print("1: Full dataset (animals)")
        print("2: Reduced dataset (mini_animals)")
        dataset_choice = input("Enter 1 or 2: ")

    subsample_dir = os.path.join(data_dir, "animals" if dataset_choice == "1" else "mini_animals")

    # --- Choose architecture ---
    architecture = ""
    while architecture.lower() not in ["vgg16", "vgg11"]:
        architecture = input("Choose model architecture to load (vgg16 or vgg11): ").lower()

    # --- Paths ---
    models_dir = Path("models")
    output_dir = Path("predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load corresponding model ---
    model_path = models_dir / f"{architecture}_state_dict.pth"

    # --- Run inference ---
    run_inference(model_path=model_path, data_dir=subsample_dir, architecture=architecture, output_path=output_dir)


if __name__ == "__main__":
    main()
