import torch
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from my_projects.dataset import AnimalsDataModule
from my_projects.modeling.train import VGGNet
import os
import argparse


def run_inference(model_path: Path, data_dir: Path, architecture: str, output_path: Path, batch_size: int = 16):
    """
    Run inference using a trained VGG model (VGG11 or VGG16).
    """
    print(f"\n🔍 Loading model: {model_path}")
    print(f"📁 Using dataset: {data_dir}")
    print(f"🧠 Architecture: {architecture}")

    # --- Setup data module ---
    data_module = AnimalsDataModule(data_dir=data_dir, batch_size=batch_size)
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


def main(
    architecture: str = "vgg16",
    dataset_choice: str = "mini",  # "full" or "mini"
    batch_size: int = 16
):
    data_dir = Path("data")
    subsample_dir = data_dir / ("animals" if dataset_choice == "full" else "mini_animals")
    models_dir = Path("models")
    output_dir = Path("predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{architecture}_state_dict.pth"
    run_inference(model_path=model_path, data_dir=subsample_dir, architecture=architecture, output_path=output_dir, batch_size=batch_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a trained VGG model for the Animals dataset.")
    parser.add_argument("--architecture", default="vgg16", choices=["vgg16", "vgg11"])
    parser.add_argument("--dataset-choice", default="mini", choices=["full", "mini"])
    parser.add_argument("--batch-size", type=int, default=16)

    args = parser.parse_args()
    main(
        architecture=args.architecture,
        dataset_choice=args.dataset_choice,
        batch_size=args.batch_size
    )