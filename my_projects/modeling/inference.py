# my_projects/modeling/inference.py

import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import numpy as np
from pathlib import Path
from my_projects.dataset import AnimalsDataModule, subsample_dir
from my_projects.modeling.train import VGGNet


def run_inference(model_path: Path, output_path: Path):
    """
    Load a trained model and generate predictions on the validation set.

    Parameters
    ----------
    model_path : Path
        Path to the saved .pth or .ckpt model file.
    output_path : Path
        Path to save the numpy arrays with predictions and labels.
    """
    # --- Setup data module ---
    data_module = AnimalsDataModule(data_dir=subsample_dir, batch_size=32)
    data_module.setup()

    val_loader = data_module.val_dataloader()
    num_classes = len(data_module.class_names)

    # --- Load model ---
    model = VGGNet(num_clases=num_classes, pretrained=False)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # --- Use GPU if available ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

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
    np.save(output_path / "val_probs.npy", all_probs)
    np.save(output_path / "val_labels.npy", all_labels)

    print(f"Saved validation predictions to {output_path}")


if __name__ == "__main__":
    models_dir = Path("models")
    output_dir = Path("predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "vgg_state_dict.pth"
    run_inference(model_path=model_path, output_path=output_dir)
