import torch
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from my_projects.dataset import AnimalsDataModule
from my_projects.modeling.train import VGGNet
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import argparse

def run_inference(model_path: Path, data_dir: Path, architecture: str, output_path: Path, batch_size: int = 16):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"⚙️  Running on: {device}")

    all_probs, all_labels = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            probs = F.softmax(out, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(yb.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # --- Save predictions ---
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / f"{architecture}_val_probs.npy", all_probs)
    np.save(output_path / f"{architecture}_val_labels.npy", all_labels)
    print(f"\n✅ Saved validation predictions to: {output_path}")

    # --- Compute simple metrics ---
    preds = all_probs.argmax(axis=1)
    acc = accuracy_score(all_labels, preds)
    cm = confusion_matrix(all_labels, preds)

    print(f"Validation Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(cm)

    # --- Save confusion matrix figure ---
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(output_path / f"{architecture}_confusion_matrix.png")
    plt.close()

    return {"val_acc": acc, "confusion_matrix": cm}
